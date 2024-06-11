# -*- coding=utf-8 -*-

import os
import json
import PIL.Image
import numpy as np
from tqdm import tqdm
from module import get_color_map, find_dir, find_img, parse_labelimg, parse_labelme, rectangle_include_point

##################################################################
#
#   此文件用于文本检测数据集转换格式
#
#   1. 从 labelimg 矩形标注裁出小图, 缩放后(可选), 保存到文件
#   2. 从 labelme 多边形标注提取文本框在小图上的相对坐标
#   3. 转为 mmocr 或 paddocr 训练所需的格式.
#
##################################################################

generate_anns_check_image = False
palette = get_color_map(80)


def generate_format_label_string(shapes, width, height, relative_path, format="paddle"):
    if format == "paddle":
        anns = [{"transcription": id[0], "points": np.asarray(shape).tolist()} for id, shape in shapes.items()]
        result = f"{relative_path}\t{json.dumps(anns, ensure_ascii=False)}\n"

    elif format == "mmlab":
        anns = []
        for instance, shape in shapes.items():
            ns_tl = np.asarray(shape).min(axis=0).tolist()
            ns_br = np.asarray(shape).max(axis=0).tolist()
            ann = {
                "iscrowd": 0,
                "category_id": 1,
                "bbox": [ns_tl[0], ns_tl[1], ns_br[0] - ns_tl[0], ns_br[1] - ns_tl[1]],
                "segmentation": np.asarray(shape).reshape(1, -1).tolist(),
                "text": instance[0],
            }
            anns.append(ann)
        result = {"file_name": relative_path, "height": height, "width": width, "annotations": anns}
        result = f"{json.dumps(result, ensure_ascii=False)}\n"

    else:
        raise Exception("Only support Paddle OCR format and mmlab OCR format")

    return result


# 单个图片
def generate(img_path, det_path, seg_path, keep_ratio, save_root, save_relative, resize=736):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size
    assert img_width > 0 and img_height > 0
    # parse anns file
    bbox = parse_labelimg(det_path, img_width, img_height)
    masks, shapes = parse_labelme(seg_path, img_width, img_height)
    centers = {instance: np.asarray(shape).mean(axis=0) for instance, shape in shapes.items()}
    # generate anns
    anns_dict = []
    for idx, (_, box) in enumerate(bbox.items()):
        # 找到所有在框内的形状, 并为这些形状添加位移和约束
        box = np.array(box)
        in_shapes = {}
        for instance, shape in shapes.items():
            if not rectangle_include_point(box, centers[instance]):
                continue
            new_shape = np.asarray(shape).reshape(-1, 2)
            for p in new_shape:
                p[0] = max(box[0], min(p[0], box[2]))
                p[1] = max(box[1], min(p[1], box[3]))
            in_shapes[instance] = new_shape - box[:2]
        if len(in_shapes) == 0:
            continue
        # crop and save crop img
        box_width = int(box[2] - box[0])
        box_height = int(box[3] - box[1])
        crop_img = img.crop(box).convert("RGB")
        if keep_ratio:
            # update params
            img_length = max(box_width, box_height)
            offset = np.array([max((img_length - box_width) // 2, 0), max((img_length - box_height) // 2, 0)])
            scale = resize / img_length if resize > img_length else 1
            box_width = resize if resize > img_length else img_length
            box_height = resize if resize > img_length else img_length
            for instance, shape in in_shapes.items():
                in_shapes[instance] = (shape + offset) * scale
            # pad and resize image
            temp = PIL.Image.new("RGB", (img_length, img_length), (0, 0, 0))
            temp.paste(crop_img, (offset[0], offset[1]))
            crop_img = temp.resize((box_width, box_height), PIL.Image.BICUBIC)
        rel_path = f"dataset/{save_relative}_{idx}.jpg"
        crop_img.save(f"{save_root}/{rel_path}")
        # in_shapes round to int
        for instance, shape in in_shapes.items():
            in_shapes[instance] = np.rint(shape).astype(int)
        # generate anns label string
        anns_dict.append(generate_format_label_string(in_shapes, box_width, box_height, rel_path))
        # generate anns check image
        if not generate_anns_check_image:
            continue
        for index, (instance, shape) in enumerate(in_shapes.items()):
            # set mask to image
            mask = masks[instance].astype(np.uint8)
            mask[mask == 0] = 255
            mask[mask == 1] = 128  # 透明度 50 %
            mask = PIL.Image.fromarray(mask, mode="L")
            mask = mask.crop(box)
            # crop mask image
            crop_mask = PIL.Image.new("L", (img_length, img_length), 255)
            crop_mask.paste(mask, (offset[0], offset[1]))
            crop_mask = crop_mask.resize((box_width, box_height), PIL.Image.BICUBIC)
            color_img = PIL.Image.new("RGB", (box_width, box_height), palette[index % 80])
            crop_img = PIL.Image.composite(crop_img, color_img, crop_mask)
        crop_img.save(f"{save_root}/dataset_viz/{save_relative}_{idx}.jpg")

    return anns_dict


def process(root_path, save_dir, split_ratio, keep_ratio):
    print(f"\n[info] start task...")
    work_path = os.path.join(root_path, "src")
    save_path = os.path.join(root_path, save_dir)
    assert os.path.isdir(work_path), f"数据集不存在: {work_path}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    dataset = []
    for dir in find_dir(work_path):
        # 获取img文件列表
        imgs_dir_path = os.path.join(work_path, dir, "imgs")
        assert os.path.isdir(imgs_dir_path), f"图片文件夹不存在: {imgs_dir_path}"
        img_list = find_img(imgs_dir_path)
        # makedirs
        os.makedirs(os.path.join(save_path, "dataset", dir), exist_ok=True)
        if generate_anns_check_image:
            os.makedirs(os.path.join(save_path, "dataset_viz", dir), exist_ok=True)
        # 遍历 ann 文件列表
        for file in tqdm(img_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN"):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f"{work_path}/{dir}/imgs/{raw_name}{extension}"
            det_path = f"{work_path}/{dir}/anns/{raw_name}.xml"
            seg_path = f"{work_path}/{dir}/anns_seg/{raw_name}.json"
            # 解析单个 ann 文件
            label_string = generate(img_path, det_path, seg_path, keep_ratio, save_path, f"{dir}/{raw_name}")
            for str in label_string:
                dataset.append(str)
    with open(f"{save_path}/all_list.txt", "w", encoding='utf-8') as file:
        file.writelines(dataset)
    test_data = dataset[::split_ratio]
    with open(f"{save_path}/test.txt", "w", encoding='utf-8') as file:
        file.writelines(test_data)
    del dataset[::split_ratio]
    with open(f"{save_path}/train.txt", "w", encoding='utf-8') as file:
        file.writelines(dataset)


if __name__ == "__main__":
    process(os.getcwd(), "dataset", 5, True)
    print("\nAll process success\n")
