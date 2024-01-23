# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import json
import PIL.Image
import collections
import xml.etree.ElementTree as ET
from tqdm import tqdm

try:
    import pycocotools.coco
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    exit(1)

detection_class = "Scale"
keypoints_class = ["beg_tl", "beg_br", "end_tl", "end_br", "point"]
skeleton = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 判断点 point 是否在矩形 rect 内部. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point(r, p):
    return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]


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
    assert os.path.isfile(xml_path), f"标签文件不存在: {xml_path}"
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
    bbox_dict = []
    for obj in getXmlValue(root, "object", 0):
        name = getXmlValue(obj, "name", 1).text
        assert name == detection_class, f'required categories: {detection_class}, this box categories: {name}'
        bndbox = getXmlValue(obj, "bndbox", 1)
        xmin = int(float(getXmlValue(bndbox, "xmin", 1).text) - 1)
        ymin = int(float(getXmlValue(bndbox, "ymin", 1).text) - 1)
        xmax = int(float(getXmlValue(bndbox, "xmax", 1).text))
        ymax = int(float(getXmlValue(bndbox, "ymax", 1).text))
        assert xmax > xmin and ymax > ymin and xmax <= image_width and ymax <= image_height, f"{xml_path}"
        bbox_dict.append([xmin, ymin, xmax, ymax])
    return bbox_dict


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_path, image_width, image_height):
    assert os.path.isfile(json_path), f"标签文件不存在: {json_path}"
    # 处理标注文件中的多余数据
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if data["imageData"] != None:
        data["imageData"] = None
        image_name = os.path.basename(data["imagePath"])
        data["imagePath"] = f"../imgs/{image_name}"
        with open(json_path, "w", encoding="utf-8") as file_out:
            file_out.write(json.dumps(data))
    # 验证图片与标签是否对应
    assert image_width == int(data["imageWidth"]), f"图片与标签不对应: {json_path}"
    assert image_height == int(data["imageHeight"]), f"图片与标签不对应: {json_path}"
    # 遍历 shapes
    shape_labels = []
    shape_points = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "point":
            name = shape["label"]
            assert name in keypoints_class, f'error point categories: {name}, Check your annotations!!!'
            shape_labels.append(name)
            shape_points.append([int(shape["points"][0][0]), int(shape["points"][0][1])])

    return shape_labels, shape_points


# 单个图片
def parse_image(img_path, xml_path, json_path, img_id, bbox_id):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelme anns file
    shape_labels, shape_points = parse_labelme(json_path, width, height)
    # parse labelimg anns file
    bbox_dict = parse_labelimg(xml_path, width, height)
    # generate anns
    imgs_dict = dict(id=img_id, file_name=img_path, width=width, height=height)
    anns_dict = []
    for idx, box in enumerate(bbox_dict):
        # 找到所有在框内的点
        bbox_keypoints_dict = {}
        for i in range(len(shape_points)):
            if rectangle_include_point(box, shape_points[i]):
                bbox_keypoints_dict[shape_labels[i]] = shape_points[i]
        # 将这些点按类别排序
        key_points = []
        for cls in keypoints_class:
            if cls in bbox_keypoints_dict:
                key_points.append(bbox_keypoints_dict[cls][0])
                key_points.append(bbox_keypoints_dict[cls][1])
                key_points.append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
            else:
                key_points.append(0)
                key_points.append(0)
                key_points.append(0)
        # 组成一个框的标签
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        annotation = dict(
            id=bbox_id + idx,
            image_id=img_id,
            category_id=1,
            bbox=[box[0], box[1], box_w, box_h],
            segmentation=[],
            area=box_w * box_h,
            num_keypoints=len(bbox_keypoints_dict),
            keypoints=key_points,
            iscrowd=0,
        )
        anns_dict.append(annotation)

    return imgs_dict, anns_dict


# 创建 coco
def process(split, all_reserver=55):
    print(f"\n[info] start task...")
    # 定义类别
    cat = {
        "id": 0,
        "name": detection_class,
        "supercategory": detection_class,
        'keypoints': keypoints_class,
        'skeleton': skeleton,
    }
    data_train = dict(categories=[cat], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[cat], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = 0
    train_bbox_id = 0
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
            json_path = f"{pre_dir}/anns_pts/{raw_name}.json"

            if split <= 0 or image_list_size < all_reserver or num % split != 0:
                imgs_dict, anns_dict = parse_image(img_path, xml_path, json_path, train_img_id, train_bbox_id)
                train_img_id += 1
                train_bbox_id += len(anns_dict)
                data_train["images"].append(imgs_dict)
                for ann in anns_dict:
                    data_train["annotations"].append(ann)

            if split <= 0 or image_list_size < all_reserver or num % split == 0:
                imgs_dict, anns_dict = parse_image(img_path, xml_path, json_path, test_img_id, test_bbox_id)
                test_img_id += 1
                test_bbox_id += len(anns_dict)
                data_test["images"].append(imgs_dict)
                for ann in anns_dict:
                    data_test["annotations"].append(ann)

            not_ann_cnt += 1 if len(anns_dict) == 0 else 0
        if not_ann_cnt != 0:
            print(f"\033[1;31m[Error] {pre_dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m")
    # 导出到文件
    with open("./pose_train.json", "w") as f:
        json.dump(data_train, f, indent=4)
    checkCOCO("./pose_train.json")  # 检查COCO文件是否正确
    with open("./pose_test.json", "w") as f:
        if len(data_test["images"]) == 0:
            data_test = data_train
        json.dump(data_test, f, indent=4)
    checkCOCO("./pose_test.json")  # 检查COCO文件是否正确


if __name__ == "__main__":
    process(0)
    print("\nAll process success\n")
