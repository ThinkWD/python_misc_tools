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
import xml.etree.ElementTree as ET

try:
    import pycocotools.coco
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.name for item in os.scandir(path) if item.is_dir()]


# 判断点 point 是否在矩形 rect 内部. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point(r, p):
    return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]


def get_color_map(num_classes):
    color_map = [(0, 0, 0) for _ in range(num_classes)]
    for i in range(num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i][0] |= ((lab >> 0) & 1) << (7 - j)
            color_map[i][1] |= ((lab >> 1) & 1) << (7 - j)
            color_map[i][2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    return color_map


# 取出 xml 内容 (length 预期长度, 为 0 则不检查)
def getXmlValue(root, name, length):
    XmlValue = root.findall(name)
    if length > 0:
        if len(XmlValue) != length:
            raise Exception("The size of %s is supposed to be %d, but is %d." % (name, length, len(XmlValue)))
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


# 解析单个 labelimg 标注文件(xml)
def parse_labelimg(det_path, img_width, img_height):
    if not os.path.isfile(det_path):
        return {}
    try:
        tree = ET.parse(det_path)
        root = tree.getroot()
        # check image size
        imgsize = getXmlValue(root, "size", 1)
        assert img_width == int(getXmlValue(imgsize, "width", 1).text), f"图片与标签不对应: {det_path}"
        assert img_height == int(getXmlValue(imgsize, "height", 1).text), f"图片与标签不对应: {det_path}"
        # parse box info
        bbox = {}
        for obj in getXmlValue(root, "object", 0):
            name = getXmlValue(obj, "name", 1).text
            bndbox = getXmlValue(obj, "bndbox", 1)
            xmin = round(float(getXmlValue(bndbox, "xmin", 1).text))
            ymin = round(float(getXmlValue(bndbox, "ymin", 1).text))
            xmax = round(float(getXmlValue(bndbox, "xmax", 1).text))
            ymax = round(float(getXmlValue(bndbox, "ymax", 1).text))
            assert xmax > xmin and ymax > ymin and xmax <= img_width and ymax <= img_height, f"{det_path}"
            bbox[(name, uuid.uuid1())] = [xmin, ymin, xmax, ymax]
    except Exception as e:
        raise Exception(f"Failed to parse XML file: {det_path}, {e}")
    return bbox


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
            raise Exception(f"Annotation ids in '{coco_file}' are not unique! duplicate items: {duplicate_items}")


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
def parse_labelme(
    seg_path,
    img_width,
    img_height,
    allow_shape_type=['circle', 'rectangle', 'line', 'linestrip', 'point', 'polygon', 'rotation'],
):
    if not os.path.isfile(seg_path):
        return {}, {}
    # load json label file
    with open(seg_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # check image size
    assert img_width == int(data["imageWidth"]), f"图片与标签不对应: {seg_path}"
    assert img_height == int(data["imageHeight"]), f"图片与标签不对应: {seg_path}"
    # parse shapes info
    masks = {}
    shapes = collections.defaultdict(list)  # 如果你访问一个不存在的键, defaultdict 会自动为这个键创建一个默认值
    for shape in data["shapes"]:
        # check shape type (rotation == polygon)
        shape_type = shape["shape_type"]
        if shape_type not in allow_shape_type:
            raise Exception(f"Unsupported shape types: {shape_type}, check: {seg_path}")
        # get instance (唯一实例 flag 值)
        label = shape["label"]
        group_id = uuid.uuid1() if shape["group_id"] is None else shape["group_id"]
        instance = (label, group_id)
        # generate mask (如果存在同一 group_id 的 mask , 就合并它们)
        points = shape["points"]
        mask = shape_to_mask([img_height, img_width], points, shape_type)
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
            points = np.stack((x, y), axis=1).flatten()
        else:
            points = np.asarray(points).flatten().tolist()
        # points round to int
        shapes[instance].append(points)
    # shapes convert to normal dict
    shapes = dict(shapes)

    return masks, shapes
