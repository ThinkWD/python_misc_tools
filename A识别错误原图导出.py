# -*- coding=utf-8 -*-

import os
import shutil
import argparse
from tqdm import tqdm

##################################################################
#
#   此文件用于根据结果文件夹得到识别错误的图片文件原图
#
#   使用步骤:
#   1. 使用节点程序识别文件夹 images, 生成结果图片到 images/result 目录
#   2. 检查 images/result 目录下的结果图片, 识别正确的删掉, 识别错误的留下
#   3. 输入参数 images 执行脚本, 将识别错误的图片的原图复制到 images/export 目录下
#
##################################################################


def process(root_path):
    assert os.path.isdir(root_path), f"图片文件夹不存在: {root_path}"
    res_path = os.path.join(root_path, "result")
    assert os.path.isdir(res_path), f"结果文件夹不存在: {res_path}"
    exp_path = os.path.join(root_path, "export")
    os.makedirs(exp_path)
    raw_img_dict = {
        name: it.name
        for it in os.scandir(root_path)
        if it.is_file() and it.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        for name, _ in [os.path.splitext(it.name)]
    }
    res_img_list = [it.name for it in os.scandir(res_path) if it.is_file() and it.name.endswith('_Final.jpg')]
    assert len(raw_img_dict) >= len(res_img_list), f"结果文件夹中的图片比原图还多"
    for image in tqdm(res_img_list, leave=True, colour="CYAN"):
        res_name = os.path.splitext(image)[0][:-6]
        if res_name in raw_img_dict:
            raw_name = raw_img_dict[res_name]
            shutil.copy(os.path.join(root_path, raw_name), os.path.join(exp_path, raw_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', type=str, help='图片文件夹绝对路径')
    args = parser.parse_args()
    process(args.root_path)
    print("\nAll process success\n")
