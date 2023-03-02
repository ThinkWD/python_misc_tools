# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import json
import xml.etree.ElementTree as ET
import imagesize
from tqdm import tqdm
from pycocotools.coco import COCO
from collections import Counter


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 遍历目录得到所有文件
def find_files(path):
    return [item.path for item in os.scandir(path) if item.is_file()]


def checkCOCO(coco_file):
    coco_api = COCO(coco_file)
    img_ids = sorted(list(coco_api.imgs.keys()))
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if "minival" not in coco_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        if len(set(ann_ids)) != len(ann_ids):
            print(
                f"\n\n\n\033[1;31m Annotation ids in '{coco_file}' are not unique!\033[0m"
            )
            result = dict(Counter(ann_ids))
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
def getXml(imgpath, xmlpath, categories, img_id, bbox_id, task, remove_categories):
    width, height = imagesize.get(imgpath)
    assert width > 0 and height > 0
    image = {
        "id": img_id,
        "file_name": imgpath,
        "width": width,
        "height": height,
        "valid": True,
        "rotate": 0,
    }
    anns = []

    if task != "val" and os.path.isfile(xmlpath):
        tree = ET.parse(xmlpath)  # 打开文件
        root = tree.getroot()  # 获取根节点
        # 框信息
        index = 0
        for obj in getXmlValue(root, "object", 0):
            # 取出 category
            category = getXmlValue(obj, "name", 1).text
            # 检查 category 是否在排除列表中
            if category not in remove_categories:
                # 更新 category 字典
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                # 得到 category_id
                category_id = categories[category]
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
                annotation = {
                    "image_id": img_id,
                    "id": bbox_id + index,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "iscrowd": 0,  # 设置分割数据，点的顺序为逆时针方向
                    "segmentation": [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]],
                    "category_id": category_id,
                    "area": o_width * o_height,
                    "order": 1,
                    # "ignore": 0,
                }
                index += 1
                anns.append(annotation)

    return image, anns


# 创建 coco
def voc_convert(root_path, task, start_img_id, start_bbox_id, remove_categories):
    json_dict = {"images": [], "annotations": [], "categories": []}  # 创建 coco 格式的基本结构
    categories = {"": 0}  # 类别
    # 获取初始索引ID
    img_id = start_img_id
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
        assert os.path.exists(img_path) and os.path.isdir(img_path)
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
                image, anns = getXml(
                    imgpath,
                    xmlpath,
                    categories,
                    img_id,
                    bbox_id,
                    task,
                    remove_categories,
                )
                # 更新索引ID
                img_id += 1
                bbox_id += len(anns)
                not_ann_cnt += 1 if len(anns) == 0 else 0
                # 更新 json_dict
                json_dict["images"].append(image)
                for ann in anns:
                    json_dict["annotations"].append(ann)
                # 更新进度条
                pbar.update(1)
        if not_ann_cnt != 0:
            print(
                "\033[1;31m",
                f"[Error] 路径{task}/{pre_dir}中有{not_ann_cnt}张图片不存在标注文件！！\n",
                "\033[0m",
            )
    # (解析xml结束)更新 categories 项
    for cate, cid in categories.items():
        cat = {"id": cid, "name": cate, "supercategory": cate}
        json_dict["categories"].append(cat)
    # 导出并保存到Json文件
    with open(f"./{task}.json", "w") as json_fp:
        json_str = json.dumps(json_dict, indent=4)
        json_fp.write(json_str)
    # 检查COCO文件是否正确
    checkCOCO(f"./{task}.json")


if __name__ == "__main__":
    # root path
    root_path = os.getcwd()
    # 图片的ID起始值
    start_img_id = 0
    # 检测框的ID起始值
    start_bbox_id = 0

    remove_categories = ["person"]

    # 根据建立的文件夹判断要进行哪些任务
    train_dir = f"{root_path}/train"
    if os.path.exists(train_dir) and os.path.isdir(train_dir):
        print("\n[info] task : train...")
        voc_convert(
            root_path,
            "train",
            start_img_id,
            start_bbox_id,
            remove_categories,
        )

    test_dir = f"{root_path}/test"
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        print("\n[info] task : test...")
        voc_convert(
            root_path,
            "test",
            start_img_id,
            start_bbox_id,
            remove_categories,
        )

    val_dir = f"{root_path}/val"
    if os.path.exists(val_dir) and os.path.isdir(val_dir):
        print("\n[info] task : val...")
        voc_convert(
            root_path,
            "val",
            start_img_id,
            start_bbox_id,
            remove_categories,
        )

    print("\nAll process success\n")
