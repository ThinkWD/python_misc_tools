# -*- coding=utf-8 -*-

import os
import numpy as np
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm
from module import (
    get_color_map,
    find_dir,
    find_img,
    parse_labelimg,
    parse_labelme,
    rectangle_include_point,
    shape_to_mask,
)


##################################################################
#
#   此文件用于生成先检测, 后语义分割的 VOC 数据集格式, 并生成 mmseg 或 paddleseg 训练所需文件
#
##################################################################

generate_anns_check_image = False
color_map = np.asarray(get_color_map(256)).flatten().tolist()


def generate(img_path, det_path, seg_path, classes, save_root, save_relative, keep_ratio, resize, format="paddle"):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size
    assert img_width > 0 and img_height > 0
    # parse anns file
    bbox = parse_labelimg(det_path, img_width, img_height)
    masks, shapes = parse_labelme(seg_path, img_width, img_height)
    shapes = {instance: shape for instance, shape in shapes.items() if instance[0] in classes}
    centers = {instance: np.asarray(shape).mean(axis=0) for instance, shape in shapes.items()}
    # generate anns
    anns_dict = []
    for idx, (_, box) in enumerate(bbox.items()):
        # 找到所有在框内的形状, 并为这些形状添加位移
        box = np.array(box)
        in_shapes = {}
        for instance, shape in shapes.items():
            if not rectangle_include_point(box, centers[instance]):
                continue
            new_shape = np.asarray(shape).reshape(-1, 2)
            in_shapes[instance] = new_shape - box[:2]
        if len(in_shapes) == 0:
            continue
        in_masks = {instance: masks[instance] for instance, _ in in_shapes.items()}
        # crop and save crop img
        box_width = int(box[2] - box[0])
        box_height = int(box[3] - box[1])
        img_length = max(box_width, box_height)
        crop_img = img.crop(box).convert("RGB")
        if keep_ratio:
            # update image
            offset = np.array([max((img_length - box_width) // 2, 0), max((img_length - box_height) // 2, 0)])
            temp = PIL.Image.new("RGB", (img_length, img_length), (0, 0, 0))
            temp.paste(crop_img, (offset[0], offset[1]))
            # set resize
            scale = resize / img_length if resize > img_length else 1
            img_length = max(resize, img_length)
            crop_img = temp.resize((img_length, img_length), PIL.Image.BICUBIC)
            # update in_shapes and in_masks
            for instance, shape in in_shapes.items():
                in_shapes[instance] = (shape + offset) * scale
                in_masks[instance] = shape_to_mask([img_length, img_length], in_shapes[instance], "polygon")
        rel_path = f"dataset/{save_relative}_{idx}"
        crop_img.save(f"{save_root}/{rel_path}.jpg")
        # crop and save crop mask
        label_mask = np.zeros((img_length, img_length), dtype=np.int8)
        for instance, mask in masks.items():
            label_mask[mask] = classes.index(instance[0])
        if label_mask.min() < 0 or label_mask.max() > 255:
            raise Exception(f'[{seg_path}] Cannot save the pixel-wise class label as PNG.')
        lbl_pil = PIL.Image.fromarray(label_mask.astype(np.uint8), mode='P')
        lbl_pil.putpalette(color_map)
        lbl_pil.save(f"{save_root}/{rel_path}.png")
        # set label string
        if format == "paddle":
            anns_dict.append(f"{rel_path}.jpg {rel_path}.png\n")
        elif format == "mmlab":
            anns_dict.append(f"{rel_path}\n")
        else:
            raise Exception("Only support Paddle OCR format and mmlab OCR format")
    return anns_dict


def process(root_path, save_dir, split_ratio, keep_ratio=True, resize=512, format="paddle"):
    print(f"\n[info] start task...")
    # init classes
    with open(f"{root_path}/classes.txt", "r", encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    assert len(classes) < 255, f"check {root_path}/classes.txt"
    assert classes[0] == '__ignore__' and classes[1] == '_background_', f"check {root_path}/classes.txt"

    work_path = os.path.join(root_path, "src")
    save_path = os.path.join(root_path, save_dir)
    assert os.path.isdir(work_path), f"数据集不存在: {work_path}"
    os.makedirs(save_path, exist_ok=True)

    # start work
    dataset = []
    for dir in find_dir(work_path):
        # 获取img文件列表
        imgs_dir_path = os.path.join(work_path, dir, "imgs")
        assert os.path.isdir(imgs_dir_path), f"图片文件夹不存在: {imgs_dir_path}"
        imgs_list = find_img(imgs_dir_path)
        # makedirs
        os.makedirs(os.path.join(save_path, "dataset", dir), exist_ok=True)
        if generate_anns_check_image:
            os.makedirs(os.path.join(save_path, "dataset_viz", dir), exist_ok=True)
        # 遍历图片列表
        for file in tqdm(imgs_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN"):
            raw_name, extension = os.path.splitext(file)
            img_path = f"{work_path}/{dir}/imgs/{raw_name}{extension}"
            det_path = f"{work_path}/{dir}/anns/{raw_name}.xml"
            seg_path = f"{work_path}/{dir}/anns_seg/{raw_name}.json"
            # generate
            strs = generate(img_path, det_path, seg_path, classes, save_path, f"{dir}/{raw_name}", keep_ratio, resize)
            dataset.extend(strs)
    with open(f"{save_path}/all_list.txt", "w", encoding='utf-8') as file:
        file.writelines(dataset)
    test_data = dataset[::split_ratio]
    with open(f"{save_path}/test.txt", "w", encoding='utf-8') as file:
        file.writelines(test_data)
    del dataset[::split_ratio]
    with open(f"{save_path}/train.txt", "w", encoding='utf-8') as file:
        file.writelines(dataset)


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == "__main__":
    process(os.getcwd(), "dataset", 5)
    print("\nAll process success\n")
