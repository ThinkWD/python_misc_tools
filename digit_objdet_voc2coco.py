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
root_path = os.getcwd()
start_imgs_id = 0  # 图片 ID 起始值
start_bbox_id = 0  # 检测框 ID 起始值
# 生成的数据集允许的标签列表
categories = ["D000", "D001", "P000", "P001"]

# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = set()


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


# 解析单个 VOC 标注文件(xml)
def getXml(imgpath, xmlpath, img_id, bbox_id):
    assert os.path.isfile(imgpath), f"图片文件不存在: {imgpath}"
    img = PIL.Image.open(imgpath)
    width, height = img.size
    assert width > 0 and height > 0
    imgs_dict = dict(
        id=img_id,
        file_name=imgpath,
        width=width,
        height=height,
    )
    anns_dict = []
    # 检查标签文件是否存在
    if not os.path.isfile(xmlpath):
        return imgs_dict, anns_dict
    # 读标签文件
    try:
        tree = ET.parse(xmlpath)  # 打开文件
        root = tree.getroot()  # 获取根节点
    except Exception:
        print(f"Failed to parse XML file: {xmlpath}")
        exit()
    # 验证宽高
    imgsize = getXmlValue(root, "size", 1)
    assert width == int(getXmlValue(imgsize, "width", 1).text), f"图片与标签不对应: {imgpath}"
    assert height == int(getXmlValue(imgsize, "height", 1).text), f"图片与标签不对应: {imgpath}"
    # 提取标签信息
    index = 0  # 框信息
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
        assert xmax > xmin and ymax > ymin
        w = xmax - xmin
        h = ymax - ymin
        annotation = dict(
            id=bbox_id + index,
            image_id=img_id,
            category_id=category_id,
            bbox=[xmin, ymin, w, h],
            segmentation=[[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]],
            area=w * h,
            iscrowd=0,
        )
        index += 1
        anns_dict.append(annotation)

    return imgs_dict, anns_dict


# 创建 coco
def voc_convert(task, split):
    print(f"\n[info] start task...")
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 获取初始索引ID
    train_img_id = start_imgs_id
    train_bbox_id = start_bbox_id
    test_img_id = 0
    test_bbox_id = 0
    # 遍历 path 下的子文件夹
    path = os.path.join(root_path, task)
    dirs = find_dir(path)
    for dir in dirs:
        not_ann_cnt = 0  # 不存在标注的文件数量
        pre_dir = os.path.basename(dir)  # 获取并打印子文件夹名
        img_path = os.path.join(dir, "imgs")  # 获取img文件列表
        assert os.path.isdir(img_path), f"图片文件夹不存在: {img_path}"
        img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 遍历XML列表
        for num, file in enumerate(tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN")):
            # 获取文件名(带多文件夹的相对路径)
            raw_name, extension = os.path.splitext(file)
            imgpath = f"{task}/{pre_dir}/imgs/{raw_name}{extension}"
            xmlpath = f"{task}/{pre_dir}/anns/{raw_name}.xml"
            if split != 0 and num % split == 0:
                # 解析 xml 文件
                image, anns = getXml(imgpath, xmlpath, test_img_id, test_bbox_id)
                # 更新索引ID
                test_img_id += 1
                test_bbox_id += len(anns)
                # 更新 json_dict
                data_test["images"].append(image)
                for ann in anns:
                    data_test["annotations"].append(ann)
            else:
                # 解析 xml 文件
                image, anns = getXml(imgpath, xmlpath, train_img_id, train_bbox_id)
                # 更新索引ID
                train_img_id += 1
                train_bbox_id += len(anns)
                # 更新 json_dict
                data_train["images"].append(image)
                for ann in anns:
                    data_train["annotations"].append(ann)
            not_ann_cnt += 1 if len(anns) == 0 else 0
        if not_ann_cnt != 0:
            print(f"\033[1;31m[Error] {task}/{pre_dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m")
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
    assert os.path.isdir(os.path.join(root_path, "src"))
    voc_convert("src", 20)
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f"\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}")
    print("\nAll process success\n")
