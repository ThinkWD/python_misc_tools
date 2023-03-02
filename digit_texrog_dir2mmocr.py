# -*- coding: UTF-8 -*-
import os
import shutil


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
                files = find_files(sec_dir)
                for file in files:
                    filename = os.path.basename(file)
                    check = filename.strip()
                    if check != filename:
                        print(f"filename: {check} 命名有问题！！！")
                        exit(0)
                    f.write(f"{pre_dir}/{label}/{filename} {label}\n")
    with open("all_list.txt", "r") as f:
        list_train = f.readlines()
    list_test = list_train[::split_ratio]
    with open("test.txt", "a") as file:
        file.writelines(list_test)
    del list_train[::split_ratio]
    with open("train.txt", "a") as file:
        file.writelines(list_train)


def txt2dir(path):
    dirs = find_dir(path)
    for dir in dirs:
        pre_dir = os.path.basename(dir)
        annfile = f"{path}/{pre_dir}/_train.txt"
        if os.path.exists(annfile):
            with open(annfile, "r") as f:
                print(f">> {pre_dir}")
                for line in f:
                    # 删除字符串两边的空格和换行
                    line = line.strip()
                    pos_split = line.find(' ')
                    file = line[line.find('/') + 1 : pos_split]
                    label = line[pos_split + 1 :]
                    # 创建文件夹
                    mkdir = f"{path}/{pre_dir}/{label}/"
                    if not os.path.exists(mkdir):
                        os.makedirs(mkdir)
                    # 移动文件
                    movefile_pre = f"{path}/{pre_dir}/{file}"
                    movefile_res = f"{path}/{pre_dir}/{label}/{file}"
                    shutil.move(movefile_pre, movefile_res)


if __name__ == "__main__":
    path = os.getcwd()
    dir2txt(path, 10)
