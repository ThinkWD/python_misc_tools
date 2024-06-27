# -*- coding=utf-8 -*-

import os
from tqdm import tqdm


##################################################################
#
#   此文件用于文本识别数据集转换格式
#
#   从文件夹获取文本内容, 然后生成文本. 此文本 mmocr 和 paddocr 通用
#
##################################################################


# 生成的数据集允许的标签列表
categories = ['#', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'L', 'U']
# 数据集中出现的在允许列表中的标签
user_categories = set()
# 数据集中出现的不在允许列表中的标签
skip_categories = set()


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


def dir2txt(root_path, split_ratio, format="paddle"):
    if format == "paddle":
        sep = '\t'
    elif format == "mmlab":
        sep = ' '
    else:
        raise Exception("Only support Paddle OCR format and mmlab OCR format")

    dataset = []
    for dir in find_dir(root_path):
        first_name = os.path.basename(dir)
        assert first_name == first_name.strip(), f"first_name: {first_name} 命名有问题！！！"
        for second in tqdm(find_dir(dir), desc=f"{first_name}\t", leave=True, ncols=100, colour="CYAN"):
            second_name = os.path.basename(second)
            assert second_name == second_name.strip(), f"second_name: {second_name} 命名有问题！！！"
            # 检查标签
            label = second_name
            for i, char in enumerate(label):
                if char in categories:
                    user_categories.add(char)
                else:
                    skip_categories.add(char)
                    label = f'{label[:i]}#{label[i + 1:]}'
            imgs_list = [f for f in os.listdir(second) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for file in imgs_list:
                basename = os.path.basename(file)
                dataset.append(f"{first_name}/{second_name}/{basename}{sep}{label}\n")

    with open(os.path.join(root_path, "all_list.txt"), "w", encoding='utf-8') as file:
        file.writelines(dataset)
    test_data = dataset[::split_ratio]
    with open(os.path.join(root_path, "test.txt"), "w", encoding='utf-8') as file:
        file.writelines(test_data)
    del dataset[::split_ratio]
    with open(os.path.join(root_path, "train.txt"), "w", encoding='utf-8') as file:
        file.writelines(dataset)
    # dict
    dict_list = sorted(user_categories, key=ord)
    with open(os.path.join(root_path, "dict_file.txt"), "w", encoding='utf-8') as file:
        for item in dict_list:
            file.write(f"{item}\n")
    print(f"\n\033[34m[Info] 出现且允许的标签列表: \033[0m{dict_list}")
    if len(skip_categories) > 0:
        print(f"\n\033[33m[Warning] 出现但不被允许的标签: \033[0m{sorted(skip_categories, key=ord)}")


if __name__ == "__main__":
    dir2txt(os.getcwd(), 10)
    print("\nAll process success\n")
