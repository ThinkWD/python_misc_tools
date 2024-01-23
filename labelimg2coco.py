# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import sys
import json
import PIL.Image
import collections
from tqdm import tqdm
import xml.etree.ElementTree as ET

try:
    import pycocotools.coco
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

# 根目录
start_imgs_id = 0  # 图片 ID 起始值
start_bbox_id = 0  # 检测框 ID 起始值
# 生成的数据集允许的标签列表
categories = ["D000", "D001", "P000", "P001"]

# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = set()


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 检查 COCO 文件是否有问题
def checkCOCO(coco_file):
    coco_api = pycocotools.coco.COCO(coco_file)
    img_ids = sorted(list(coco_api.imgs.keys()))
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if "minival" not in coco_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        if len(set(ann_ids)) != len(ann_ids):
            print(f"\n\n\n\033[1;31m Annotation ids in '{coco_file}' are not unique!\033[0m")
            result = dict(collections(ann_ids))
            # print(result)
            # print([key for key, value in result.items() if value > 1])
            print({key: value for key, value in result.items() if value > 1})
            print("\n\n\n")
            exit()


# 取出 xml 内容 (length 预期长度，为 0 则不检查)
def getXmlValue(root, name, length):
    # root为xml文件的根节点，name是子节点，作用为取出子节点内容
    XmlValue = root.findall(name)
    # 检查取出的值长度是否符合预期; 0 不检查
    if len(XmlValue) == 0:
        raise NotImplementedError(f"Can not find {name} in {root.tag}.")
    if length > 0:
        if len(XmlValue) != length:
            raise NotImplementedError("The size of %s is supposed to be %d, but is %d." % (name, length, len(XmlValue)))
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


# 解析单个 labelimg 标注文件(xml)
def parse_labelimg(xml_path, image_width, image_height):
    if not os.path.isfile(xml_path):
        return [], []
    # 读标签文件
    try:
        tree = ET.parse(xml_path)  # 打开文件
        root = tree.getroot()  # 获取根节点
    except Exception:
        print(f"Failed to parse XML file: {xml_path}")
        exit()
    # 验证图片与标签是否对应
    imgsize = getXmlValue(root, "size", 1)
    assert image_width == int(getXmlValue(imgsize, "width", 1).text), f"图片与标签不对应: {xml_path}"
    assert image_height == int(getXmlValue(imgsize, "height", 1).text), f"图片与标签不对应: {xml_path}"
    # 提取框信息
    lable_dict = []
    bbox_dict = []
    for obj in getXmlValue(root, "object", 0):
        # 取出 category
        category = getXmlValue(obj, "name", 1).text
        # 检查 category 是否在允许列表中
        if category not in categories:
            skip_categories.add(category)
            continue
        category_id = categories.index(category)  # 得到 category_id
        # 取出框
        bndbox = getXmlValue(obj, "bndbox", 1)
        xmin = int(float(getXmlValue(bndbox, "xmin", 1).text) - 1)
        ymin = int(float(getXmlValue(bndbox, "ymin", 1).text) - 1)
        xmax = int(float(getXmlValue(bndbox, "xmax", 1).text))
        ymax = int(float(getXmlValue(bndbox, "ymax", 1).text))
        assert xmax > xmin and ymax > ymin and xmax <= image_width and ymax <= image_height, f"{xml_path}"
        lable_dict.append(category_id)
        bbox_dict.append([xmin, ymin, xmax, ymax])
    return lable_dict, bbox_dict


# 单个图片
def parse_image(img_path, xml_path, img_id, bbox_id):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelimg anns file
    lable_dict, bbox_dict = parse_labelimg(xml_path, width, height)
    # generate anns
    imgs_dict = dict(id=img_id, file_name=img_path, width=width, height=height)
    anns_dict = []
    for idx, box in enumerate(bbox_dict):
        # 组成一个框的标签
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        annotation = dict(
            id=bbox_id + idx,
            image_id=img_id,
            category_id=lable_dict[idx],
            bbox=[box[0], box[1], box_w, box_h],
            segmentation=[],
            area=box_w * box_h,
            iscrowd=0,
        )
        anns_dict.append(annotation)
    return imgs_dict, anns_dict


def voc2coco(split, all_reserver=55):
    print(f"\n[info] start task...")
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = start_imgs_id
    train_bbox_id = start_bbox_id
    test_img_id = 0
    test_bbox_id = 0
    # 遍历脚本所在目录下的子文件夹
    for dir in find_dir(os.getcwd()):
        not_ann_cnt = 0  # 不存在标注的文件数量
        pre_dir = os.path.basename(dir)  # 获取并打印子文件夹名
        img_path = os.path.join(dir, "imgs")  # 获取img文件列表
        assert os.path.isdir(img_path), f"图片文件夹不存在: {img_path}"
        img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 遍历XML列表
        image_list_size = len(img_list)
        for num, file in enumerate(tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN")):
            # 获取文件名(带多文件夹的相对路径)
            raw_name, extension = os.path.splitext(file)
            img_path = f"{pre_dir}/imgs/{raw_name}{extension}"
            xml_path = f"{pre_dir}/anns/{raw_name}.xml"

            if split <= 0 or image_list_size < all_reserver or num % split != 0:
                imgs_dict, anns_dict = parse_image(img_path, xml_path, train_img_id, train_bbox_id)
                train_img_id += 1
                train_bbox_id += len(anns_dict)
                data_train["images"].append(imgs_dict)
                for ann in anns_dict:
                    data_train["annotations"].append(ann)

            if split <= 0 or image_list_size < all_reserver or num % split == 0:
                imgs_dict, anns_dict = parse_image(img_path, xml_path, test_img_id, test_bbox_id)
                test_img_id += 1
                test_bbox_id += len(anns_dict)
                data_test["images"].append(imgs_dict)
                for ann in anns_dict:
                    data_test["annotations"].append(ann)

            not_ann_cnt += 1 if len(anns_dict) == 0 else 0
        if not_ann_cnt != 0:
            print(f"\033[1;31m[Error] {pre_dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m")
    # 导出到文件
    for id, category in enumerate(categories):
        cat = {"id": id, "name": category, "supercategory": category}
        data_train["categories"].append(cat)  # 训练集
        data_test["categories"].append(cat)  # 测试集
    with open("./train.json", "w") as f:
        json.dump(data_train, f, indent=4)
    checkCOCO("./train.json")  # 检查COCO文件是否正确
    with open("./test.json", "w") as f:
        json.dump(data_test, f, indent=4)
    checkCOCO("./test.json")  # 检查COCO文件是否正确


if __name__ == "__main__":
    voc2coco(20)
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f"\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}")
    print("\nAll process success\n")
