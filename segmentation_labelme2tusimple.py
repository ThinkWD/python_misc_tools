# -*- coding=utf-8 -*-

import os
import json
import PIL.Image
import numpy as np
from tqdm import tqdm
from module import find_dir, find_img, parse_labelme


##################################################################
#
#   此文件用于语义分割线段标注转为车道线检测 tusimple 格式
#
#   一, 将在文件夹名上标注了 '_R' 后缀的相关文件执行旋转操作, 结果保存到 '_rotated' 文件夹下
#   二, 将 test.txt, train.txt 内部的 '_R' 替换为 '_rotated'
#   三, 从 test.txt, train.txt 生成 tusimple 标注文件
#
##################################################################


def data_shapes_rotate(shapes, width, height):
    x_scale = height / width
    y_scale = width / height
    for shape in shapes:
        for pt in shape['points']:
            x_tmp = width - pt[0]
            pt[0] = pt[1] * y_scale
            pt[1] = x_tmp * x_scale


def process_rotate(root_path):
    imgs_path = os.path.join(root_path, "imgs")
    anns_path = os.path.join(root_path, "anns_seg")
    pngs_path = os.path.join(root_path, "anns_png")
    assert os.path.isdir(imgs_path), "imgs directory not exists."
    assert os.path.isdir(anns_path), "anns_seg directory not exists."
    assert os.path.isdir(pngs_path), "anns_png directory not exists."
    # rotate image
    for dir in find_dir(imgs_path):
        if not dir.endswith('_R'):
            continue
        # 获取img文件列表
        imgs_dir_path = os.path.join(imgs_path, dir)
        assert os.path.isdir(imgs_dir_path), f"图片文件夹不存在: {imgs_dir_path}"
        imgs_list = find_img(imgs_dir_path)
        # makedirs
        prefix = f"{dir[:-1]}rotated"
        os.makedirs(os.path.join(imgs_path, prefix), exist_ok=True)
        os.makedirs(os.path.join(anns_path, prefix), exist_ok=True)
        os.makedirs(os.path.join(pngs_path, prefix), exist_ok=True)
        # 遍历图片列表
        for file in tqdm(imgs_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN"):
            raw_name, extension = os.path.splitext(file)
            img_path = f"{imgs_path}/{dir}/{raw_name}{extension}"
            seg_path = f"{anns_path}/{dir}/{raw_name}.json"
            png_path = f"{pngs_path}/{dir}/{raw_name}.png"
            # rotate img
            cur_image = PIL.Image.open(img_path)
            width, height = cur_image.size
            cur_image = cur_image.rotate(90, expand=True)
            cur_image = cur_image.resize((width, height), PIL.Image.BICUBIC)
            cur_image.save(f'{imgs_path}/{prefix}/{raw_name}{extension}')
            # rotate png
            cur_image = PIL.Image.open(png_path)
            width, height = cur_image.size
            cur_image = cur_image.rotate(90, expand=True)
            cur_image = cur_image.resize((width, height), PIL.Image.BICUBIC)
            cur_image.save(f'{pngs_path}/{prefix}/{raw_name}.png')
            # rotate seg
            with open(seg_path, "r", encoding='utf-8') as file:
                data = json.load(file)
            data_shapes_rotate(data['shapes'], width, height)
            data["imagePath"] = f"../../imgs/{prefix}/{raw_name}{extension}"
            with open(f"{anns_path}/{prefix}/{raw_name}.json", "w", encoding='utf-8') as file:
                json.dump(data, file, indent=4)


def line_to_line_intersect(lineA, lineB, cross=0):
    if cross == 0:
        cross = lineA[2] * lineB[3] - lineB[2] * lineA[3]
        if abs(cross) < 1e-6:
            return [-1, -1]
    t = (lineB[3] * (lineB[0] - lineA[0]) - lineB[2] * (lineB[1] - lineA[1])) / cross
    return [lineA[0] + t * lineA[2], lineA[1] + t * lineA[3]]


def generate_tusimple(img_path, seg_path, relative_path):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size
    assert img_width > 0 and img_height > 0
    _, shapes = parse_labelme(seg_path, img_width, img_height, ['line', 'linestrip'])
    # init tusimple
    tusimple = dict(lanes=[], h_samples=[], raw_file=relative_path)
    lanes = []
    h_samples = list(range(round(img_height * 0.1 / 4.5) * 10, img_height, 10))
    for _, shape in shapes.items():
        points = np.asarray(shape).reshape(-1, 2)
        points_length = len(points)
        # check monotonic
        y_diff = np.diff(points[:, 1])
        monotonic = 0 if np.all(y_diff > 0) else (1 if np.all(y_diff < 0) else 2)
        assert monotonic == 0 or monotonic == 1, f"非法标注: {seg_path}"
        # init lane
        lane = np.full((len(h_samples),), -2, dtype=float)
        for i in range(points_length - 1):
            # y range
            line_y_l = points[i][1] if monotonic == 0 else points[i + 1][1]
            line_y_h = points[i][1] if monotonic == 1 else points[i + 1][1]
            # line
            lineA = [points[i][0], points[i][1], points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]]
            lineB = [0, 0, img_width, 0]
            # cross
            cross = lineA[2] * lineB[3] - lineB[2] * lineA[3]
            if abs(cross) < 1e-6:
                continue
            # intersect
            for idx, h in enumerate(h_samples):
                if h > line_y_h or h < line_y_l:
                    continue
                lineB[1] = h
                pt = line_to_line_intersect(lineA, lineB, cross)
                if pt[0] >= 0 and pt[0] <= img_width and pt[1] >= line_y_l and pt[1] <= line_y_h:
                    lane[idx] = pt[0]
        lanes.append(lane.tolist())
    tusimple['h_samples'] = h_samples
    tusimple['lanes'] = lanes
    return tusimple


def process_tusimple(root_path, label_file):
    imgs_path = os.path.join(root_path, "imgs")
    anns_path = os.path.join(root_path, "anns_seg")
    assert os.path.isdir(imgs_path), "imgs directory not exists."
    assert os.path.isdir(anns_path), "anns_seg directory not exists."
    # get test list
    with open(os.path.join(root_path, label_file), "r") as f:
        label_list = f.readlines()
    tusimples = []
    for idx, str in enumerate(tqdm(label_list, leave=True, ncols=100, colour="CYAN")):
        path = str[: str.find(' ')]
        assert path.startswith('imgs/'), f"check {label_file}"
        assert PIL.Image.open(os.path.join(imgs_path, path[5:])).size == (1920, 1080), "All image must be 1920 * 1080!"
        # get path info
        prefix, fullname = os.path.split(os.path.normpath(path[5:]))
        raw_name, extension = os.path.splitext(fullname)
        # update prefix and label_list
        prefix = f"{prefix[:-1]}rotated" if prefix.endswith('_R') else prefix
        label_list[idx] = f"imgs/{prefix}/{raw_name}{extension} anns_png/{prefix}/{raw_name}.png\n"
        # gen tusimple
        img_path = f"{imgs_path}/{prefix}/{raw_name}{extension}"
        seg_path = f"{anns_path}/{prefix}/{raw_name}.json"
        tusimple = generate_tusimple(img_path, seg_path, f"imgs/{prefix}/{raw_name}{extension}")
        tusimples.append(json.dumps(tusimple))
    # export tusimples
    raw_name, _ = os.path.splitext(label_file)
    os.makedirs(f"{root_path}/{raw_name}_set", exist_ok=True)
    with open(f"{root_path}/{raw_name}_set/{raw_name}_list.txt", "w") as file:
        file.writelines(label_list)
    with open(f"{root_path}/{raw_name}_set/{raw_name}_label.json", "w") as file:
        file.write("\n".join(tusimples))


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == "__main__":
    root_path = os.getcwd()
    process_rotate(root_path)
    process_tusimple(root_path, "test.txt")
    process_tusimple(root_path, "train.txt")
