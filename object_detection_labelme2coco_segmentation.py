# -*- coding=utf-8 -*-

import os
import json
import PIL.Image
import PIL.ImageDraw
import numpy as np
import pycocotools.mask
from tqdm import tqdm
from module import get_color_map, find_dir, find_img, parse_labelme, checkCOCO


##################################################################
#
#   此文件用于 实例分割/旋转框 数据集转换格式, 从 labelme 多边形标注转为 COCO 格式
#
##################################################################


# 生成的数据集允许的标签列表
categories = ["switch"]
# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = []
palette = get_color_map(80)


def generate(img_path, seg_path, viz_path=""):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelme anns file
    masks, shapes = parse_labelme(seg_path, width, height)
    # generate anns
    imgs_dict = dict(id=0, file_name=img_path, width=width, height=height)
    anns_dict = []
    for instance, mask in masks.items():
        label = instance[0]
        if label not in categories:
            if label not in skip_categories:
                skip_categories.append(label)
            continue
        label_id = categories.index(label)
        a_mask = np.asfortranarray(mask.astype(np.uint8))  # 将 mask 转为 Fortran 无符号整数数组
        a_mask = pycocotools.mask.encode(a_mask)  # 将 mask 编码为 RLE 格式
        area = float(pycocotools.mask.area(a_mask))  # 计算 mask 的面积
        bbox = pycocotools.mask.toBbox(a_mask).flatten().tolist()  # 计算边界框(x,y,w,h)
        annotation = dict(
            id=0,
            image_id=0,
            category_id=label_id,
            bbox=bbox,
            segmentation=shapes[instance],
            area=area,
            iscrowd=0,
        )
        anns_dict.append(annotation)
        if len(viz_path) == 0:
            continue
        # draw mask
        color = palette[label_id % 80]  # 获取颜色
        mask = mask.astype(np.uint8)
        mask[mask == 0] = 255
        mask[mask == 1] = 128
        mask = PIL.Image.fromarray(mask, mode="L")
        color_img = PIL.Image.new("RGB", mask.size, color)
        img = PIL.Image.composite(img, color_img, mask)
        # draw bbox
        bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=color, width=2)
    if len(anns_dict) > 0 and len(viz_path) > 0:
        img.save(viz_path)
    return imgs_dict, anns_dict


def process(root_path, split, all_reserve=0, reserve_no_label=False):
    print(f"\n[info] start task...")
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = 0
    train_bbox_id = 0
    test_img_id = 0
    test_bbox_id = 0
    for dir in find_dir(root_path):
        # vizs
        vizs_dir_path = os.path.join(root_path, f"viz_{dir}")
        os.makedirs(vizs_dir_path, exist_ok=True)
        # img_list
        imgs_dir_path = os.path.join(root_path, dir, "imgs")
        assert os.path.isdir(imgs_dir_path), f"图片文件夹不存在: {imgs_dir_path}"
        img_list = find_img(imgs_dir_path)
        all_reserve_dir = len(img_list) < all_reserve
        not_ann_cnt = 0
        for num, file in enumerate(tqdm(img_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN")):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f"{dir}/imgs/{raw_name}{extension}"
            seg_path = f"{dir}/anns_seg/{raw_name}.json"
            viz_path = f"{vizs_dir_path}/{raw_name}.jpg"
            # get dict
            imgs_dict, anns_dict = generate(img_path, seg_path, viz_path)
            # check anns_dict size
            anns_size = len(anns_dict)
            not_ann_cnt += 1 if anns_size == 0 else 0
            if reserve_no_label == False and anns_size == 0:
                continue
            # train dataset
            if all_reserve_dir or split <= 0 or num % split != 0:
                imgs_dict["id"] = train_img_id
                data_train["images"].append(imgs_dict.copy())
                for idx, ann in enumerate(anns_dict):
                    ann["image_id"] = train_img_id
                    ann["id"] = train_bbox_id + idx
                    data_train["annotations"].append(ann.copy())
                train_img_id += 1
                train_bbox_id += anns_size
            # test dataset
            if all_reserve_dir or split <= 0 or num % split == 0:
                imgs_dict["id"] = test_img_id
                data_test["images"].append(imgs_dict.copy())
                for idx, ann in enumerate(anns_dict):
                    ann["image_id"] = test_img_id
                    ann["id"] = test_bbox_id + idx
                    data_test["annotations"].append(ann.copy())
                test_img_id += 1
                test_bbox_id += anns_size
        if not_ann_cnt != 0:
            print(f"\033[1;31m[Warning] {dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m")

    print(f"\n训练集图片总数: {train_img_id}, 标注总数: {train_bbox_id}\n")
    print(f"测试集图片总数: {test_img_id}, 标注总数: {test_bbox_id}\n")
    # export to file
    for id, category in enumerate(categories):
        cat = {"id": id, "name": category, "supercategory": category}
        data_train["categories"].append(cat)
        data_test["categories"].append(cat)
    with open("./train.json", "w", encoding='utf-8') as f:
        json.dump(data_train, f, indent=4)
    checkCOCO("./train.json")
    with open("./test.json", "w", encoding='utf-8') as f:
        json.dump(data_test, f, indent=4)
    checkCOCO("./test.json")


if __name__ == "__main__":
    process(os.getcwd(), 10)
    if len(skip_categories) > 0:
        print(f"\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}")
    print("\nAll process success\n")
