import os
import json
import shutil
import PIL.Image
import numpy as np
from tqdm import tqdm


##################################################################
#
#   此文件用于语义分割数据集转换格式, 从 labelme 线段标注转为 tusimple 格式, 并生成训练所需 test_label.json 文件
#
##################################################################
supported_shape_type = ['line', 'linestrip']


def line_to_line_intersect(lineA, lineB, cross=0):
    if cross == 0:
        cross = lineA[2] * lineB[3] - lineB[2] * lineA[3]
        if abs(cross) < 1e-6:
            return [-1, -1]
    t = (lineB[3] * (lineB[0] - lineA[0]) - lineB[2] * (lineB[1] - lineA[1])) / cross
    return [lineA[0] + t * lineA[2], lineA[1] + t * lineA[3]]


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_path):
    assert os.path.isfile(json_path), f"文件不存在: {json_path}"
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    tusimple = dict(lanes=[], h_samples=[], raw_file='')
    # raw_file
    raw_file = data['imagePath']
    tusimple['raw_file'] = raw_file[raw_file.find('imgs/') :]
    # h_samples
    imageHeight = data['imageHeight']
    imageWidth = data['imageWidth']
    h_samples = list(range(round(imageHeight * 0.1 / 4.5) * 10, imageHeight, 10))
    # lanes
    lanes = []
    shapes = data['shapes']
    for shape in shapes:
        assert shape['shape_type'] in supported_shape_type, f"不支持的形状: {shape['shape_type']}, {json_path}"
        points = np.array(shape['points'], dtype=float)
        points_length = len(points)
        assert points_length > 1, f"点数太少: {points_length}, {json_path}"
        y_diff = np.diff(points[:, 1])
        monotonic = 0 if np.all(y_diff > 0) else (1 if np.all(y_diff < 0) else 2)
        assert monotonic == 0 or monotonic == 1, f"非法标注: {json_path}"
        # init lane
        lane = np.full((len(h_samples),), -2, dtype=float)
        for i in range(points_length - 1):
            # y range
            line_y_l = points[i][1] if monotonic == 0 else points[i + 1][1]
            line_y_h = points[i][1] if monotonic == 1 else points[i + 1][1]
            # line
            lineA = [points[i][0], points[i][1], points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]]
            lineB = [0, 0, imageWidth, 0]
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
                if pt[0] >= 0 and pt[0] <= imageWidth and pt[1] >= line_y_l and pt[1] <= line_y_h:
                    lane[idx] = pt[0]
        lanes.append(lane.tolist())
    tusimple['h_samples'] = h_samples
    tusimple['lanes'] = lanes
    return tusimple


def get_path_information(path):
    prefix, fullname = os.path.split(os.path.normpath(path))
    name, format = os.path.splitext(fullname)
    return {'prefix': prefix, 'name': name, 'format': format}


def main(root_path, label_file):
    anns_prefix = os.path.join(root_path, "anns_seg")
    assert os.path.isdir(anns_prefix), "anns_seg directory not exists."
    assert os.path.isdir(os.path.join(root_path, "imgs")), "imgs directory not exists."
    # get test list
    with open(os.path.join(root_path, label_file), "r") as f:
        label_list = f.readlines()
    tusimples = []
    for str in tqdm(label_list, leave=True, ncols=100, colour="CYAN"):
        img_path = str[: str.find(' ')]
        assert PIL.Image.open(img_path).size == (1920, 1080), "All image must be 1920 * 1080!"
        path_info = get_path_information(img_path[5:])
        anns_path = os.path.join(anns_prefix, path_info['prefix'], f"{path_info['name']}.json")
        tusimples.append(json.dumps(parse_labelme(anns_path)))
    # export tusimples
    with open(os.path.join(root_path, "test_label.json"), "w") as file:
        file.write("\n".join(tusimples))


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == "__main__":
    root_path = os.getcwd()
    main(root_path, "test.txt")
    # organizational directory structure
    os.makedirs(f"{root_path}/test_set", exist_ok=True)
    os.makedirs(f"{root_path}/train_set", exist_ok=True)
    shutil.move(f"{root_path}/train.txt", f"{root_path}/train_set/train_list.txt")
    shutil.move(f"{root_path}/test.txt", f"{root_path}/test_set/test_list.txt")
    shutil.move(f"{root_path}/test_label.json", f"{root_path}/test_set/test_label.json")
