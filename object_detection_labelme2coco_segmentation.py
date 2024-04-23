# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import sys
import json
import math
import uuid
import collections
import PIL.Image
import PIL.ImageDraw
import numpy as np
from tqdm import tqdm


##################################################################
#
#   此文件用于实例分割数据集转换格式, 从 labelme 多边形标注转为 COCO 格式
#
#   COCO 格式用于 实例分割训练, VOC 格式用于 语义分割训练
#
##################################################################


try:
    import pycocotools.coco
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


# 根目录
root_path = os.getcwd()
# 生成的数据集允许的标签列表
categories = ["person", "bottle", "chair", "sofa", "bus", "car"]
# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = []


# fmt: off
palette = [ # 来自COCO的80类调色盘
    (220, 20, 60),(119, 11, 32),(0, 0, 142),(0, 0, 230),(106, 0, 228),
    (0, 60, 100),(0, 80, 100),(0, 0, 70),(0, 0, 192),(250, 170, 30),
    (100, 170, 30),(220, 220, 0),(175, 116, 175),(250, 0, 30),(165, 42, 42),
    (255, 77, 255),(0, 226, 252),(182, 182, 255),(0, 82, 0),(120, 166, 157),
    (110, 76, 0),(174, 57, 255),(199, 100, 0),(72, 0, 118),(255, 179, 240),
    (0, 125, 92),(209, 0, 151),(188, 208, 182),(0, 220, 176),(255, 99, 164),
    (92, 0, 73),(133, 129, 255),(78, 180, 255),(0, 228, 0),(174, 255, 243),
    (45, 89, 255),(134, 134, 103),(145, 148, 174),(255, 208, 186),(197, 226, 255),
    (171, 134, 1),(109, 63, 54),(207, 138, 255),(151, 0, 95),(9, 80, 61),
    (84, 105, 51),(74, 65, 105),(166, 196, 102),(208, 195, 210),(255, 109, 65),
    (0, 143, 149),(179, 0, 194),(209, 99, 106),(5, 121, 0),(227, 255, 205),
    (147, 186, 208),(153, 69, 1),(3, 95, 161),(163, 255, 0),(119, 0, 170),
    (0, 182, 199),(0, 165, 120),(183, 130, 88),(95, 32, 0),(130, 114, 135),
    (110, 129, 133),(166, 74, 118),(219, 142, 185),(79, 210, 114),(178, 90, 62),
    (65, 70, 15),(127, 167, 115),(59, 105, 106),(142, 108, 45),(196, 172, 0),
    (95, 54, 80),(128, 76, 255),(201, 57, 1),(246, 0, 122),(191, 162, 208)
]
# fmt: on


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 遍历目录得到所有文件
def find_files(path):
    return [item.path for item in os.scandir(path) if item.is_file()]


# 检查 COCO 文件是否有问题
def checkCOCO(coco_file):
    coco_api = pycocotools.coco.COCO(coco_file)
    img_ids = sorted(list(coco_api.imgs.keys()))
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if "minival" not in coco_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        if len(set(ann_ids)) != len(ann_ids):
            result = dict(collections(ann_ids))
            duplicate_items = {key: value for key, value in result.items() if value > 1}
            raise Exception(f"Annotation ids in '{coco_file}' are not unique! duplicate items:\n{duplicate_items}")


# shape_to_mask
def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


# 保存 labelme 支持的形状类型
labelme_shape_type = ["circle", "rectangle", "line", "linestrip", "point", "polygon"]


def parse_labelme_ann(imgpath, viz_imgpath, annpath, img_id, bbox_id):
    # 读图返回图片信息
    assert os.path.isfile(imgpath), f"图片文件不存在: {imgpath}"
    img = PIL.Image.open(imgpath)
    width, height = img.size
    assert width > 0 and height > 0
    imgs_dict = dict(id=img_id, file_name=imgpath, width=width, height=height)
    anns_dict = []
    if not os.path.isfile(annpath):
        return imgs_dict, anns_dict
    # 读标签文件
    with open(annpath, "r") as file_in:
        anndata = json.load(file_in)
    assert width == anndata["imageWidth"] and height == anndata["imageHeight"], f"图片与标签不对应: {imgpath}"
    # 遍历形状
    masks = {}  # 用于存储每个实例的掩码
    segmentations = collections.defaultdict(list)  # 用于存储每个实例的分割点坐标
    for shape in anndata["shapes"]:
        # 检查 label 是否在允许列表中
        label = shape["label"]
        if label not in categories:
            if label not in skip_categories:
                skip_categories.append(label)
            continue
        # 检查形状类型
        shape_type = shape["shape_type"]
        if shape_type not in labelme_shape_type:
            continue
        # 生成mask
        points = shape["points"]
        mask = shape_to_mask([height, width], points, shape_type)
        group_id = uuid.uuid1() if shape["group_id"] is None else shape["group_id"]
        instance = (label, group_id)  # 唯一实例 flag 值
        # 如果存在同一 group_id 的 mask , 就合并它们
        masks[instance] = masks[instance] | mask if instance in masks else mask
        # 处理点集, 根据形状类型将点转换为一维列表.
        if shape_type == "rectangle":  # 矩形将两个对角点转换为四个顶点
            (x1, y1), (x2, y2) = points
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            points = [x1, y1, x2, y1, x2, y2, x1, y2]
        if shape_type == "circle":  # 圆形根据圆心和半径，生成一个多边形的点坐标。
            (x1, y1), (x2, y2) = points
            r = np.linalg.norm([x2 - x1, y2 - y1])
            # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
            # x: tolerance of the gap between the arc and the line segment
            n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
            i = np.arange(n_points_circle)
            x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
            y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
            points = np.stack((x, y), axis=1).flatten().tolist()
        else:
            points = np.asarray(points).flatten().tolist()
        segmentations[instance].append(points)
    segmentations = dict(segmentations)

    for index, (instance, mask) in enumerate(masks.items()):
        label, group_id = instance
        label_id = categories.index(label)
        a_mask = np.asfortranarray(mask.astype(np.uint8))  # 将 mask 转为 Fortran 无符号整数数组
        a_mask = pycocotools.mask.encode(a_mask)  # 将 mask 编码为 RLE 格式
        area = float(pycocotools.mask.area(a_mask))  # 计算 mask 的面积
        bbox = pycocotools.mask.toBbox(a_mask).flatten().tolist()  # 计算边界框(x,y,w,h)
        annotation = dict(
            id=bbox_id + index,
            image_id=img_id,
            category_id=label_id,
            bbox=bbox,
            segmentation=segmentations[instance],
            area=area,
            iscrowd=0,
        )
        anns_dict.append(annotation)
        # 绘图
        color = palette[index % 80]  # 获取颜色
        # 画mask
        mask = mask.astype(np.uint8)
        mask[mask == 0] = 255
        mask[mask == 1] = 128
        mask = PIL.Image.fromarray(mask, mode="L")
        color_img = PIL.Image.new("RGB", mask.size, color)
        img = PIL.Image.composite(img, color_img, mask)
        # 画框
        bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])  # bbox
        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=color, width=2)

    img.save(viz_imgpath)
    return imgs_dict, anns_dict


def labelme_convert(task):
    print(f"\n[info] task : {task}...")
    data = dict(categories=[], images=[], annotations=[])
    # 初始索引ID
    img_id = 0
    bbox_id = 0
    # 遍历 path 下的子文件夹
    path = os.path.join(root_path, task)
    dirs = find_dir(path)
    for dir in dirs:
        not_ann_cnt = 0
        # 获取并打印子文件夹名
        pre_dir = os.path.basename(dir)
        # 获取img文件列表
        img_path = os.path.join(dir, "imgs")
        assert os.path.isdir(img_path)
        img_list = os.listdir(img_path)
        # 自适应选定ann文件夹名
        anndir = "anns_seg" if os.path.isdir(f"{task}/{pre_dir}/anns_seg") else "anns"
        # 创建 viz 验证图片的文件夹
        if not os.path.isdir(f"{task}_viz/{pre_dir}"):
            os.makedirs(f"{task}_viz/{pre_dir}")
        # 遍历img文件列表
        for file in tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
            # 获取文件名(带多文件夹的相对路径)
            file = file.strip()
            imgpath = f"{task}/{pre_dir}/imgs/{file}"
            viz_imgpath = f"{task}_viz/{pre_dir}/{file}"
            annpath = f"{task}/{pre_dir}/{anndir}/{file[:file.rindex('.')]}.json"
            # 解析 ann 文件
            imgs, anns = parse_labelme_ann(imgpath, viz_imgpath, annpath, img_id, bbox_id)
            # 更新索引ID
            img_id += 1
            bbox_id += len(anns)
            not_ann_cnt += 1 if len(anns) == 0 else 0
            # 更新 json_data
            data["images"].append(imgs)
            for ann in anns:
                data["annotations"].append(ann)
        if not_ann_cnt != 0:
            print(f"\033[1;31m[Error] 路径{task}/{pre_dir}中有{not_ann_cnt}张图片不存在标注文件！！\n\033[0m")
    # 更新 categories 项
    for id, category in enumerate(categories):
        cat = dict(id=id, name=category, supercategory=category)
        data["categories"].append(cat)
    # 导出并保存到json文件
    with open(f"./{task}.json", "w") as f:
        json.dump(data, f, indent=4)
    # 检查COCO文件是否正确
    checkCOCO(f"./{task}.json")


if __name__ == "__main__":
    # 根据建立的文件夹判断要进行哪些任务
    if os.path.isdir(f"{root_path}/train"):
        labelme_convert("train")
    if os.path.isdir(f"{root_path}/test"):
        labelme_convert("test")
    if os.path.isdir(f"{root_path}/val"):
        labelme_convert("val")
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f"\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}")
    print("\nAll process success\n")
