# -*- coding: UTF-8 -*-
import os


##################################################################
#
#   此文件用于文本识别数据集转换格式
#
#   从文件夹获取文本内容, 然后生成文本. 此文本 mmocr 和 paddocr 通用
#
##################################################################


# 生成的数据集允许的标签列表
categories = ['#', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'C', 'D']
# 数据集中出现的在允许列表中的标签
user_categories = set()
# 数据集中出现的不在允许列表中的标签
skip_categories = set()


def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


def find_files(path):
    return [item.path for item in os.scandir(path) if item.is_file()]


def dir2txt(path, split_ratio):
    dirs = find_dir(path)
    with open("all_list.txt", "a") as f:
        for dir in dirs:
            pre_dir = os.path.basename(dir)
            check = pre_dir.strip()
            print(f"pre_dir: {check}")
            if check != pre_dir:
                print(f"pre_dir: {check} 命名有问题！！！")
                exit(0)
            sec_dirs = find_dir(dir)
            for sec_dir in sec_dirs:
                label = os.path.basename(sec_dir)
                check = label.strip()
                if check != label:
                    print(f"label: {check} 命名有问题！！！")
                    exit(0)
                # 检查标签
                label_bak = label
                for i, char in enumerate(label):
                    if char in categories:
                        user_categories.add(char)
                    else:
                        skip_categories.add(char)
                        label = f'{label[:i]}#{label[i + 1:]}'
                # 遍历写入文件
                files = find_files(sec_dir)
                for file in files:
                    basename = os.path.basename(file)
                    name, exten = os.path.splitext(basename)
                    if not name.isdigit() or exten.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                        print(f"\n\033[33m[Warning] 非图片文件: \033[0m{file}")
                        continue
                    f.write(f"{pre_dir}/{label_bak}/{basename} {label}\n")
    with open("all_list.txt", "r") as f:
        list_train = f.readlines()
    list_test = list_train[::split_ratio]
    with open("test.txt", "a") as file:
        file.writelines(list_test)
    del list_train[::split_ratio]
    with open("train.txt", "a") as file:
        file.writelines(list_train)


if __name__ == "__main__":
    path = os.getcwd()
    assert (
        not os.path.isfile("all_list.txt")
        and not os.path.isfile("test.txt")
        and not os.path.isfile("train.txt")
        and not os.path.isfile("dict_file.txt")
    )
    dir2txt(path, 10)
    dict_list = sorted(user_categories, key=ord)
    with open("dict_file.txt", "a") as file:
        for item in dict_list:
            file.write(f"{item}\n")
    print(f"\n\033[34m[Info] 出现且允许的标签列表: \033[0m{dict_list}")
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f"\n\033[33m[Warning] 出现但不被允许的标签: \033[0m{sorted(skip_categories, key=ord)}")
    print("\nAll process success\n")
