# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import json
import sys
import xml.etree.ElementTree as ET
import imagesize
from tqdm import tqdm
import collections

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
categories = ["D000", "D001"]  # 数字仪表
# allow_list = ["P000", "P001"]  # 指针仪表

# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = []


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
            print(
                f"\n\n\n\033[1;31m Annotation ids in '{coco_file}' are not unique!\033[0m"
            )
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
            raise NotImplementedError(
                "The size of %s is supposed to be %d, but is %d."
                % (name, length, len(XmlValue))
            )
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


# 解析单个 VOC 标注文件(xml)
def getXml(imgpath, xmlpath, img_id, bbox_id, task):
    width, height = imagesize.get(imgpath)
    assert width > 0 and height > 0
    imgs_dict = dict(
        id=img_id,
        file_name=imgpath,
        width=width,
        height=height,
    )
    anns_dict = []
    # 检查task与文件是否存在
    if task == "val" or not os.path.isfile(xmlpath):
        return imgs_dict, anns_dict
    # 读标签文件
    tree = ET.parse(xmlpath)  # 打开文件
    root = tree.getroot()  # 获取根节点
    index = 0  # 框信息
    for obj in getXmlValue(root, "object", 0):
        # 取出 category
        category = getXmlValue(obj, "name", 1).text
        # 检查 category 是否在允许列表中
        if category not in categories:
            if category not in skip_categories:
                skip_categories.append(category)
            continue
        category_id = categories.index(category)  # 得到 category_id
        # 取出框
        bndbox = getXmlValue(obj, "bndbox", 1)
        xmin = int(getXmlValue(bndbox, "xmin", 1).text) - 1
        ymin = int(getXmlValue(bndbox, "ymin", 1).text) - 1
        xmax = int(getXmlValue(bndbox, "xmax", 1).text)
        ymax = int(getXmlValue(bndbox, "ymax", 1).text)
        assert xmax > xmin
        assert ymax > ymin
        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        annotation = dict(
            id=bbox_id + index,
            image_id=img_id,
            category_id=category_id,
            bbox=[xmin, ymin, o_width, o_height],
            segmentation=[[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]],
            area=o_width * o_height,
            iscrowd=0,  # 设置分割数据，点的顺序为逆时针方向
        )
        index += 1
        anns_dict.append(annotation)

    return imgs_dict, anns_dict


# 创建 coco
def voc_convert(task):
    data = dict(categories=[], images=[], annotations=[])  # 创建 coco 格式的基本结构
    # 获取初始索引ID
    img_id = start_imgs_id
    bbox_id = start_bbox_id
    # 遍历 path 下的子文件夹
    path = os.path.join(root_path, task)
    dirs = find_dir(path)
    for dir in dirs:
        not_ann_cnt = 0
        # 获取并打印子文件夹名
        pre_dir = os.path.basename(dir)
        # 获取img文件列表
        img_path = os.path.join(dir, "imgs")
        assert os.path.isdir(img_path), f"图片文件夹不存在: {img_path}"
        img_list = os.listdir(img_path)
        # 设置 tqdm 进度条
        with tqdm(
            total=len(img_list),  # 迭代总数
            desc=f"{pre_dir}\t",  # 进度条最前面的描述
            leave=True,  # 进度条走完是否保留
            ncols=100,  # 进度条长度
            colour="CYAN",  # 颜色(BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)
        ) as pbar:
            # 遍历XML列表
            for file in img_list:
                # 获取文件名(带多文件夹的相对路径)
                file = file.strip()
                imgpath = f"{task}/{pre_dir}/imgs/{file}"
                xmlpath = f"{task}/{pre_dir}/anns/{file[:file.rindex('.')]}.xml"
                # 解析 xml 文件
                image, anns = getXml(imgpath, xmlpath, img_id, bbox_id, task)
                # 更新索引ID
                img_id += 1
                bbox_id += len(anns)
                not_ann_cnt += 1 if len(anns) == 0 else 0
                # 更新 json_dict
                data["images"].append(image)
                for ann in anns:
                    data["annotations"].append(ann)
                # 更新进度条
                pbar.update(1)
        if not_ann_cnt != 0:
            print(
                "\033[1;31m",
                f"[Error] 路径{task}/{pre_dir}中有{not_ann_cnt}张图片不存在标注文件！！\n",
                "\033[0m",
            )
    # (解析xml结束)更新 categories 项
    for id, category in enumerate(categories):
        cat = {"id": id, "name": category, "supercategory": category}
        data["categories"].append(cat)
    # 导出并保存到Json文件
    with open(f"./{task}.json", "w") as f:
        json.dump(data, f, indent=4)
    # 检查COCO文件是否正确
    checkCOCO(f"./{task}.json")
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f"\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}")


if __name__ == "__main__":
    # 根据建立的文件夹判断要进行哪些任务
    if os.path.isdir(f"{root_path}/train"):
        print("\n[info] task : train...")
        voc_convert("train")
    if os.path.isdir(f"{root_path}/test"):
        print("\n[info] task : test...")
        voc_convert("test")
    if os.path.isdir(f"{root_path}/val"):
        print("\n[info] task : val...")
        voc_convert("val")
    print("\nAll process success\n")
