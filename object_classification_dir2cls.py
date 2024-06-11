# -*- coding=utf-8 -*-

import os
import numpy as np
from tqdm import tqdm


##################################################################
#
#   此文件用于图片分类数据集转换格式
#
#   从文件夹名获取类名, one-hot 编码, 不支持复合类别.
#
##################################################################


# 生成的数据集允许的标签列表
categories = ['not_uniform', 'blue_uniform', 'pink_uniform', 'white_uniform']
# 数据集中出现的不在允许列表中的标签
skip_categories = set()


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


def dir2cls(root_path, split_ratio):
    all_list = []
    # 第一层路径
    for first_dir in find_dir(root_path):
        # 获取类名, 检查类名, 生成编码
        categorie = os.path.basename(first_dir)
        if categorie not in categories:
            skip_categories.add(categorie)
            continue
        one_hot = np.zeros(len(categories), dtype=int)  # 生成全 0 数组
        one_hot[categories.index(categorie)] = 1  # 按索引置 1
        one_hot = ','.join([str(item) for item in one_hot])  # 将数组转为字符串
        # 遍历第二层路径
        for second_dir in find_dir(first_dir):
            second_name = os.path.basename(second_dir)
            img_list = [f for f in os.listdir(second_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for file in tqdm(img_list, desc=f"{categorie}/{second_name}\t", leave=True, ncols=150, colour="CYAN"):
                all_list.append(f"{categorie}/{second_name}/{file}\t{one_hot}\n")

    with open(os.path.join(root_path, "all_list.txt"), "w", encoding='utf-8') as file:
        file.writelines(all_list)
    test_list = all_list[::split_ratio]
    with open(os.path.join(root_path, "test.txt"), "w", encoding='utf-8') as file:
        file.writelines(test_list)
    del all_list[::split_ratio]
    with open(os.path.join(root_path, "train.txt"), "w", encoding='utf-8') as file:
        file.writelines(all_list)


if __name__ == "__main__":
    dir2cls(os.getcwd(), 10)
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f"\n\033[33m[Warning] 出现但不被允许的标签: \033[0m{sorted(skip_categories, key=ord)}")
    print("\nAll process success\n")
