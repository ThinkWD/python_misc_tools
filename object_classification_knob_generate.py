# -*- coding=utf-8 -*-

import os
import PIL.Image
import PIL.ImageOps
from tqdm import tqdm


##################################################################
#
#   此文件用于生成方向分类数据集 (先将所有图片整理为同一方向, 然后生成其它方向的图片)
#
#   默认方向为 `向上`, 不支持复合类别.
#
##################################################################


def find_dir(path):
    return [item.name for item in os.scandir(path) if item.is_dir()]


def find_img(path):
    return [
        item.name
        for item in os.scandir(path)
        if item.is_file() and item.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]


def generate(img_path, img_name, save_root, save_relative):
    # 未翻转
    flip_0 = os.path.join(save_relative, "flip_0", img_name)
    image = PIL.Image.open(os.path.join(img_path, img_name))
    image.save(os.path.join(save_root, flip_0))
    # 沿 x 轴翻转
    flip_1 = os.path.join(save_relative, "flip_1", img_name)
    temp = PIL.ImageOps.mirror(image)
    temp.save(os.path.join(save_root, flip_1))
    # 沿 y 轴翻转
    flip_2 = os.path.join(save_relative, "flip_2", img_name)
    temp = PIL.ImageOps.flip(image)
    temp.save(os.path.join(save_root, flip_2))
    # 同时沿 x, y 轴翻转
    flip_3 = os.path.join(save_relative, "flip_3", img_name)
    temp = image.transpose(PIL.Image.ROTATE_180)
    temp.save(os.path.join(save_root, flip_3))
    # get label string
    return [f"{flip_0}\t1,0\n", f"{flip_1}\t1,0\n", f"{flip_2}\t0,1\n", f"{flip_3}\t0,1\n"]


def process(root_path, save_dir, split_ratio):
    print(f"\n[info] start task...")

    work_path = os.path.join(root_path, "src")
    save_path = os.path.join(root_path, save_dir)
    assert os.path.isdir(work_path), f"数据集不存在: {work_path}"
    os.makedirs(save_path, exist_ok=True)

    # start work
    dataset = []
    for dir in find_dir(work_path):
        # 获取img文件列表
        imgs_dir_path = os.path.join(work_path, dir, "imgs_knob")
        if not os.path.isdir(imgs_dir_path):
            continue
        imgs_list = find_img(imgs_dir_path)
        # makedirs
        save_relative = os.path.join("dataset", dir)
        os.makedirs(os.path.join(save_path, save_relative, "flip_0"))  # 不翻转
        os.makedirs(os.path.join(save_path, save_relative, "flip_1"))  # 沿 x 轴翻转
        os.makedirs(os.path.join(save_path, save_relative, "flip_2"))  # 沿 y 轴翻转
        os.makedirs(os.path.join(save_path, save_relative, "flip_3"))  # 同时沿 x, y 轴翻转
        # 遍历图片列表
        for file in tqdm(imgs_list, desc=f"{dir}\t", leave=True, ncols=100, colour="CYAN"):
            dataset.extend(generate(imgs_dir_path, file, save_path, save_relative))
    with open(f"{save_path}/all_list.txt", "w", encoding='utf-8') as file:
        file.writelines(dataset)
    test_data = dataset[::split_ratio]
    with open(f"{save_path}/test.txt", "w", encoding='utf-8') as file:
        file.writelines(test_data)
    del dataset[::split_ratio]
    with open(f"{save_path}/train.txt", "w", encoding='utf-8') as file:
        file.writelines(dataset)


if __name__ == "__main__":
    process(os.getcwd(), "dataset_clas", 5)
    print("\nAll process success\n")
