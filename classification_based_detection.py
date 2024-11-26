# -*- coding=utf-8 -*-

import os
import cv2
import copy
import scipy
import numpy as np
from tqdm import tqdm
from module import find_img, parse_labelimg


##################################################################
#
#   此文件用于联合人体检测模型和安全帽检测模型结果，生成用于人体检测模型后续步骤的分类数据集
#
#   要求的文件结构：
#       root_path
#           - imgs        (完整原图)
#           - anns        (安全帽检测结果文件夹 XML)
#           - anns_person (人体检测结果文件夹 XML)
#
#   工作原理：匹配同一张图的两个检测结果，如果有安全帽在人体范围内，则按安全帽类别给此人体分类，否则丢弃此人体。
#   注意事项：
#       - 前半部分正常密度图片：不限制最大框数量, 人体检测使用 30 置信度 0 最小宽度
#       - 后半部分人员密集图片：限制最大框数量为 5, 人体检测使用 50 置信度 24 最小宽度
#
##################################################################

cats = {"黄": "yellow", "黑": "black", "红": "red", "白": "white", "橙": "orange", "蓝": "blue", "head": "head"}


# 计算包含得分 (0-1)
def compute_contain_score(person, helmet):
    # 完全包含
    if person[0] <= helmet[0] and person[1] <= helmet[1] and person[2] >= helmet[2] and person[3] >= helmet[3]:
        return 1.0
    # 部分包含 (将两框重叠面积占小框的比值作为分数)
    if person[2] >= helmet[0] and helmet[2] >= person[0] and person[3] >= helmet[1] and helmet[3] >= person[1]:
        x1 = max(person[0], helmet[0])
        y1 = max(person[1], helmet[1])
        x2 = min(person[2], helmet[2])
        y2 = min(person[3], helmet[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        helmet_area = (helmet[2] - helmet[0]) * (helmet[3] - helmet[1])
        return intersection / helmet_area
    return 0.0


# 计算距离得分 (0-1)
def compute_distance_score(person, helmet):
    # 人体框宽高
    preson_w = person[2] - person[0]
    preson_h = person[3] - person[1]
    # x 距离
    person_x = person[0] + preson_w * 0.5
    helmet_x = (helmet[0] + helmet[2]) * 0.5
    distance_x = person_x - helmet_x
    # y 距离
    person_y = person[1] + preson_h * 0.4  # 对于人脸的特殊偏上中心点
    helmet_y = (helmet[1] + helmet[3]) * 0.5
    distance_y = person_y - helmet_y
    # 实际距离
    distance = (distance_x**2 + distance_y**2) ** 0.5
    max_distance = ((preson_w * 0.5) ** 2 + (preson_h * 0.5) ** 2) ** 0.5
    return 1 - distance / max_distance


def bipartite_matching(person_boxes, helmet_boxes, score_threshold=0.6):
    """
    使用二部图建模和匈牙利算法进行全局最优匹配
    :param person_boxes: 人体框列表
    :param helmet_boxes: 人脸框列表
    :param score_threshold: 匹配分数阈值
    :return: 匹配结果
    """
    person_list = [person for _, person in person_boxes.items()]
    helmet_list = [(cats[label], helmet) for (label, _), helmet in helmet_boxes.items()]
    # 先构造包含关系的 cost 矩阵
    person_size, helmet_size = len(person_list), len(helmet_list)
    cost_matrix = np.zeros((person_size, helmet_size))
    for i, person in enumerate(person_list):
        for j, (_, helmet) in enumerate(helmet_list):
            cost_matrix[i, j] = compute_contain_score(person, helmet)
    # 筛选 cost 矩阵, 禁用类别不统一的人体框
    for i in range(person_size):
        label_list = []
        for j in range(helmet_size):
            if cost_matrix[i, j] > 0.45:  # 如果有 45% 的面积都在人体框内部，视为有效框
                label_list.append(helmet_list[j][0])
        if len(set(label_list)) > 1:  # 如果有效框的类别不统一，禁用该人体框
            cost_matrix[i, :] = 0  # 将该行清零
    # 继续添加距离关系的 cost 权重
    for i, person in enumerate(person_list):
        for j, (_, helmet) in enumerate(helmet_list):
            if cost_matrix[i, j] > 0:
                cost_matrix[i, j] = -1 * (cost_matrix[i, j] + compute_distance_score(person, helmet))
    # 匈牙利算法求解
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    # 提取匹配结果
    pairs = []
    for i, j in zip(row_ind, col_ind):
        score = -1 * cost_matrix[i, j]
        if score > score_threshold:  # 过滤掉低于阈值的匹配
            pairs.append((person_list[i], helmet_list[j][1], helmet_list[j][0], score))
    # 保留分数最高的前五个结果 (针对人群特别密集的图片)
    # if len(pairs) > 5:
    #     pairs = sorted(pairs, key=lambda x: x[3], reverse=True)[:5]
    return pairs


def get_wh_box(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def main(root_path, save_dir, debug_match=False):
    print(f"\n[info] start task...")
    imgs_path = os.path.join(root_path, 'imgs')
    anns_helmet_path = os.path.join(root_path, 'anns')
    anns_person_path = os.path.join(root_path, 'anns_person')
    save_path = os.path.join(root_path, save_dir)
    # 创建结果文件夹
    for _, label in cats.items():
        os.makedirs(os.path.join(save_path, label), exist_ok=True)
    # 开始遍历处理
    save_index = 0
    imgs = find_img(imgs_path)
    for file in tqdm(imgs, leave=True, ncols=100, colour="CYAN"):
        raw_name, _ = os.path.splitext(file)
        helmet_path = os.path.join(anns_helmet_path, f'{raw_name}.xml')
        person_path = os.path.join(anns_person_path, f'{raw_name}.xml')
        if not os.path.isfile(helmet_path) or not os.path.isfile(person_path):
            continue
        image = cv2.imread(os.path.join(imgs_path, file))
        img_height, img_width, _ = image.shape
        helmet_boxes = parse_labelimg(helmet_path, img_width, img_height, False)
        person_boxes = parse_labelimg(person_path, img_width, img_height, False)
        pairs = bipartite_matching(person_boxes, helmet_boxes)
        # debug
        if debug_match:
            for _, box in helmet_boxes.items():
                cv2.rectangle(image, get_wh_box(box), (255, 0, 0), 1)
            for person_box, helmet_box, label, _ in pairs:
                show = copy.deepcopy(image)
                cv2.rectangle(show, get_wh_box(helmet_box), (0, 0, 255), 1)
                cv2.rectangle(show, get_wh_box(person_box), (0, 0, 255), 1)
                cv2.imshow(label, show)
                cv2.waitKey(0)
        for box, _, label, _ in pairs:
            img = image[box[1] : box[3], box[0] : box[2]]
            cv2.imwrite(os.path.join(save_path, label, f'{save_index:06d}.jpg'), img)
            save_index += 1


if __name__ == "__main__":
    main(os.getcwd(), "dataset_cls")
    print("\nAll process success\n")
