# -*- coding=utf-8 -*-

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
#   此文件用于 实例分割/旋转框 数据集转换格式, 从 labelme 多边形标注转为 COCO 格式
#
##################################################################


try:
    import pycocotools.coco
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


# 生成的数据集允许的标签列表
categories = ["switch", "person", "bottle", "chair", "sofa", "bus", "car"]
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
    return [item.name for item in os.scandir(path) if item.is_dir()]


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


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_path, image_width, image_height):
    assert os.path.isfile(json_path), f"标签文件不存在: {json_path}"
    # load json label file
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # check image size
    assert image_width == int(data["imageWidth"]), f"图片与标签不对应: {json_path}"
    assert image_height == int(data["imageHeight"]), f"图片与标签不对应: {json_path}"
    # parse shapes info
    masks = {}
    segmentations = collections.defaultdict(list)  # 如果你访问一个不存在的键, defaultdict 会自动为这个键创建一个默认值
    for shape in data["shapes"]:
        # check label
        label = shape["label"]
        if label not in categories:
            if label not in skip_categories:
                skip_categories.append(label)
            continue
        # check shape type (rotation == polygon)
        shape_type = shape["shape_type"]
        if shape_type not in ['circle', 'rectangle', 'line', 'linestrip', 'point', 'polygon', 'rotation']:
            raise Exception(f"Unsupported shape types: {shape_type}, check: {json_path}")
        # get instance (唯一实例 flag 值)
        group_id = uuid.uuid1() if shape["group_id"] is None else shape["group_id"]
        instance = (label, group_id)
        # generate mask (如果存在同一 group_id 的 mask , 就合并它们)
        points = shape["points"]
        mask = shape_to_mask([image_height, image_width], points, shape_type)
        masks[instance] = masks[instance] | mask if instance in masks else mask
        # points convert
        if shape_type == "rectangle":  # 矩形将两个对角点转换为四个顶点
            (x1, y1), (x2, y2) = points
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            points = [x1, y1, x2, y1, x2, y2, x1, y2]
        elif shape_type == "circle":  # 圆形根据圆心和半径，生成一个多边形的点坐标。
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
    # segmentations convert to normal dict
    segmentations = dict(segmentations)

    return masks, segmentations


def generate(img_path, ann_path, viz_path=""):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelme anns file
    masks, segmentations = parse_labelme(ann_path, width, height)
    # generate anns
    imgs_dict = dict(id=0, file_name=img_path, width=width, height=height)
    anns_dict = []
    for instance, mask in masks.items():
        label, _ = instance
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
            segmentation=segmentations[instance],
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

    if len(viz_path) > 0:
        img.save(viz_path)
    return imgs_dict, anns_dict


def process(root_path, split, all_reserve=0, reserve_no_label=True):
    print(f"\n[info] start task...")
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = 0
    train_bbox_id = 0
    test_img_id = 0
    test_bbox_id = 0
    for dir in find_dir(root_path):
        # imgs
        imgs_dir_path = os.path.join(root_path, dir, "imgs")
        assert os.path.isdir(imgs_dir_path), f"图片文件夹不存在: {imgs_dir_path}"
        # vizs
        vizs_dir_path = os.path.join(root_path, dir, "anns_viz")
        os.makedirs(os.path.join(root_path, dir, "anns_viz"), exist_ok=True)
        # img_list
        img_list = [f for f in os.listdir(imgs_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        all_reserve_dir = len(img_list) < all_reserve
        not_ann_cnt = 0
        for num, file in enumerate(tqdm(img_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN")):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f"{dir}/imgs/{raw_name}{extension}"
            ann_path = f"{dir}/anns_seg/{raw_name}.json"
            viz_path = f"{vizs_dir_path}/{raw_name}.jpg"
            # get dict
            imgs_dict, anns_dict = generate(img_path, ann_path, viz_path)
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
    with open("./train.json", "w") as f:
        json.dump(data_train, f, indent=4)
    checkCOCO("./train.json")
    with open("./test.json", "w") as f:
        json.dump(data_test, f, indent=4)
    checkCOCO("./test.json")


if __name__ == "__main__":
    process(os.getcwd(), 10)
    if len(skip_categories) > 0:
        print(f"\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}")
    print("\nAll process success\n")
