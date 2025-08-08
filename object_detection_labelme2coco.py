import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from module import checkCOCO, find_dir, find_img, parse_labelme

##################################################################
#
#   此文件用于目标检测数据集转换格式, 从 VOC 格式转为 COCO 格式
#
##################################################################

# 生成的数据集允许的标签列表
categories = ['rect']

# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = set()


def downsample_list(lst, n):
    if n <= 0 or len(lst) <= n:
        return lst
    step = len(lst) / n
    indices = [round(i * step) for i in range(n)]
    return [lst[index] for index in indices]


def isA4Rectangle(contour, epsilon=0.15):
    """
    检查轮廓是否为符合 A4 比例的矩形
    :param contour: 输入轮廓
    :param epsilon: 允许的宽高比容差
    :return: 布尔值表示是否符合
    """
    # 多边形逼近
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon=0.02 * peri, closed=True)

    # 检查是否为凸四边形
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return False

    # 获取最小外接旋转矩形
    rect = cv2.minAreaRect(approx)
    width, height = rect[1]

    # 确保 width 为长边
    actual_width = max(width, height)
    actual_height = min(width, height)

    # 计算宽高比并验证
    aspect_ratio = actual_width / actual_height
    a4_ratio = 297.0 / 210.0  # A4 标准比例
    return abs(aspect_ratio - a4_ratio) <= epsilon


def findOuterRectangle(frame):
    """
    在图像中查找符合 A4 比例的最大外轮廓
    :param frame: 输入图像 (BGR 格式)
    :return: 外接矩形 (x, y, w, h)
    """
    # 图像预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # 寻找最大符合条件轮廓
    max_area = 0
    best_contour = None
    hierarchy = hierarchy[0]  # 解包外层数组

    for i, cnt in enumerate(contours):
        # 检查是否为根轮廓 (无父轮廓)
        if isA4Rectangle(cnt):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_contour = cnt
    assert max_area > 0
    return cv2.boundingRect(best_contour)


# 单个图片
def generate(img_path, det_path, save_path):
    # check image
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = cv2.imread(img_path)
    # 获取 img 的宽高
    height, width = img.shape[:2]
    assert width > 0 and height > 0
    # 获取目标区域
    cropped_x, cropped_y, cropped_w, cropped_h = findOuterRectangle(img)
    cropped = img[cropped_y : cropped_y + cropped_h, cropped_x : cropped_x + cropped_w]
    cv2.imwrite(save_path, cropped)

    # parse labelimg anns file
    _, shapes = parse_labelme(det_path, width, height, ['rectangle'])
    # generate anns
    imgs_dict = dict(id=0, file_name=save_path, width=cropped_w, height=cropped_h)
    anns_dict = []
    for instance, shape in shapes.items():
        label = instance[0]
        if label not in categories:
            print(f'\n{img_path}')
            skip_categories.add(label)
            continue
        label_id = categories.index(label)

        box = [shape[0][0][0], shape[0][0][1], shape[0][2][0], shape[0][2][1]]
        # 调整边界框坐标到裁剪后的坐标系
        new_xmin = max(0, box[0] - cropped_x)
        new_ymin = max(0, box[1] - cropped_y)
        new_xmax = min(cropped_w, box[2] - cropped_x)
        new_ymax = min(cropped_h, box[3] - cropped_y)

        # 跳过完全在裁剪区域外的标注
        if new_xmin >= cropped_w or new_ymin >= cropped_h or new_xmax <= 0 or new_ymax <= 0:
            continue

        # 只包含部分在裁剪区域内的标注
        if new_xmin < 0 or new_ymin < 0 or new_xmax > cropped_w or new_ymax > cropped_h:
            # 裁剪标注到有效区域
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(cropped_w, new_xmax)
            new_ymax = min(cropped_h, new_ymax)

        # 计算裁剪后的边界框尺寸
        box_w = new_xmax - new_xmin
        box_h = new_ymax - new_ymin

        annotation = dict(
            id=0,
            image_id=0,
            category_id=label_id,
            bbox=[new_xmin, new_ymin, box_w, box_h],
            segmentation=[],
            area=box_w * box_h,
            iscrowd=0,
        )
        anns_dict.append(annotation)
    return imgs_dict, anns_dict


def process(root_path, split, max_images=0, all_reserve=0, reserve_no_label=False):
    print('\n[info] start task...')
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = 0
    train_bbox_id = 0
    test_img_id = 0
    test_bbox_id = 0
    # 遍历脚本所在目录下的子文件夹
    for dir in find_dir(root_path):
        imgs_dir_path = os.path.join(root_path, dir, 'imgs')
        if not os.path.isdir(imgs_dir_path):
            continue
        os.makedirs(f'{dir}/imgs_crop', exist_ok=True)
        img_list = downsample_list(find_img(imgs_dir_path), max_images)
        all_reserve_dir = len(img_list) < all_reserve
        not_ann_cnt = 0
        for num, file in enumerate(tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN')):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f'{dir}/imgs/{raw_name}{extension}'
            det_path = f'{dir}/anns/{raw_name}.json'
            save_path = f'{dir}/imgs_crop/{raw_name}.jpg'
            # 解析获取图片和标签字典
            imgs_dict, anns_dict = generate(img_path, det_path, save_path)
            # 无标注文件计数
            anns_size = len(anns_dict)
            not_ann_cnt += 1 if anns_size == 0 else 0
            if reserve_no_label is False and anns_size == 0:
                continue
            # train dataset
            if all_reserve_dir or split <= 0 or num % split != 0:
                imgs_dict['id'] = train_img_id
                data_train['images'].append(imgs_dict.copy())
                for idx, ann in enumerate(anns_dict):
                    ann['image_id'] = train_img_id
                    ann['id'] = train_bbox_id + idx
                    data_train['annotations'].append(ann.copy())
                train_img_id += 1
                train_bbox_id += anns_size
            # test dataset
            if all_reserve_dir or split <= 0 or num % split == 0:
                imgs_dict['id'] = test_img_id
                data_test['images'].append(imgs_dict.copy())
                for idx, ann in enumerate(anns_dict):
                    ann['image_id'] = test_img_id
                    ann['id'] = test_bbox_id + idx
                    data_test['annotations'].append(ann.copy())
                test_img_id += 1
                test_bbox_id += anns_size
        if not_ann_cnt != 0:
            print(f'\033[1;31m[Error] {dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m')
    print(f'\n训练集图片总数: {train_img_id}, 标注总数: {train_bbox_id}\n')
    print(f'测试集图片总数: {test_img_id}, 标注总数: {test_bbox_id}\n')
    # 导出到文件
    for id, category in enumerate(categories):
        cat = {'id': id, 'name': category, 'supercategory': category}
        data_train['categories'].append(cat)  # 训练集
        data_test['categories'].append(cat)  # 测试集
    with open('./train.json', 'w', encoding='utf-8') as f:
        json.dump(data_train, f, indent=4)
    checkCOCO('./train.json')  # 检查COCO文件是否正确
    with open('./test.json', 'w', encoding='utf-8') as f:
        json.dump(data_test, f, indent=4)
    checkCOCO('./test.json')  # 检查COCO文件是否正确


if __name__ == '__main__':
    process(os.getcwd(), 10)
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f'\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}')
    print('\nAll process success\n')
