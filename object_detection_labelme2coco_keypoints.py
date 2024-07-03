# -*- coding=utf-8 -*-

import os
import json
import PIL.Image
from tqdm import tqdm
from module import find_dir, find_img, parse_labelimg, parse_labelme, checkCOCO, rectangle_include_point

##################################################################
#
#   此文件用于关键点检测数据集转换格式, 从 labelme 多边形标注转为 COCO 格式, 用于关键点检测训练
#
##################################################################

detection_class = "Scale"
keypoints_class = ["beg_tl", "beg_br", "end_tl", "end_br", "point"]
skeleton = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]


# 单个图片
def generate(img_path, det_path, seg_path):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelimg anns file
    bbox = parse_labelimg(det_path, width, height)
    bbox = {instance: box for instance, box in bbox.items() if instance[0] == detection_class}
    # parse labelme anns file
    _, shapes = parse_labelme(seg_path, width, height, ['point'])
    shapes = {instance: shape for instance, shape in shapes.items() if instance[0] in keypoints_class}
    # generate anns
    imgs_dict = dict(id=0, file_name=img_path, width=width, height=height)
    anns_dict = []
    for _, box in bbox.items():
        # 找到所有在框内的点 (同类别会覆盖)
        points = {instance[0]: shape[0] for instance, shape in shapes.items() if rectangle_include_point(box, shape[0])}
        # 将这些点按类别排序
        key_points = []
        for cls in keypoints_class:  # 2-可见不遮挡 1-遮挡 0-没有点
            key_points.extend([points[cls][0], points[cls][1], 2] if cls in points else [0, 0, 0])
        # 组成一个框的标签
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        annotation = dict(
            id=0,
            image_id=0,
            category_id=1,
            bbox=[box[0], box[1], box_w, box_h],
            segmentation=[],
            area=box_w * box_h,
            num_keypoints=len(points),
            keypoints=key_points,
            iscrowd=0,
        )
        anns_dict.append(annotation)
    return imgs_dict, anns_dict


# 创建 coco
def process(root_path, split, all_reserve=0, reserve_no_label=False):
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
    for dir in find_dir(root_path):
        imgs_dir_path = os.path.join(root_path, dir, "imgs")
        if not os.path.isdir(imgs_dir_path):
            continue
        img_list = find_img(imgs_dir_path)
        all_reserve_dir = len(img_list) < all_reserve
        not_ann_cnt = 0
        for num, file in enumerate(tqdm(img_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN")):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f"{dir}/imgs/{raw_name}{extension}"
            det_path = f"{dir}/anns/{raw_name}.xml"
            seg_path = f"{dir}/anns_seg/{raw_name}.json"
            # get dict
            imgs_dict, anns_dict = generate(img_path, det_path, seg_path)
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
            print(f"\033[1;31m[Error] {dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m")
    print(f"\n训练集图片总数: {train_img_id}, 标注总数: {train_bbox_id}\n")
    print(f"测试集图片总数: {test_img_id}, 标注总数: {test_bbox_id}\n")
    # export to file
    with open("./pose_train.json", "w", encoding='utf-8') as f:
        json.dump(data_train, f, indent=4)
    checkCOCO("./pose_train.json")  # 检查COCO文件是否正确
    with open("./pose_test.json", "w", encoding='utf-8') as f:
        json.dump(data_test, f, indent=4)
    checkCOCO("./pose_test.json")  # 检查COCO文件是否正确


if __name__ == "__main__":
    process(os.getcwd(), 10)
    print("\nAll process success\n")
