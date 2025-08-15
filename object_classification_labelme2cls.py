import json
import os

import numpy as np
from tqdm import tqdm

##################################################################
#
#   此文件用于分类数据集转格式, 从 labelme 格式转为 训练所需 格式
#
##################################################################

# 生成的数据集允许的标签列表
categories = ['helmet', 'smoke', 'playphone']
categories_dict = {
    '戴安全帽        (必须戴在头上)': 'helmet',
    '抽烟              (拿在手里就算)': 'smoke',
    '打电话/玩手机 (拿在手里就算)': 'playphone',
    '无效图片        (看不到头部)': 'invalid',
}


def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


def find_img(path):
    return [
        item.name
        for item in os.scandir(path)
        if item.is_file() and item.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]


def query_labelme_flags(json_path, categories):
    with open(json_path, encoding='utf-8') as file:
        data = json.load(file)
    flags = data.get('flags', {})
    return {cat: flags.get(raw) for raw, cat in categories.items()}


def process(root_path, split_ratio):
    all_list = []
    # 第一层路径
    for first_dir in find_dir(root_path):
        # 获取类名, 检查类名, 生成编码
        dirname = os.path.basename(first_dir)
        # 遍历第二层路径
        for second_dir in find_dir(first_dir):
            second_name = os.path.basename(second_dir)
            img_list = [f for f in os.listdir(second_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for file in tqdm(img_list, desc=f'{dirname}/{second_name}\t', leave=True, ncols=150, colour='CYAN'):
                raw_name, _ = os.path.splitext(file)
                json_path = f'{dirname}/{second_name}/{raw_name}.json'
                if not os.path.exists(json_path):
                    one_hot = ','.join([str(item) for item in np.zeros(len(categories), dtype=int)])
                else:
                    rdict = query_labelme_flags(f'{dirname}/{second_name}/{raw_name}.json', categories_dict)
                    if rdict['invalid']:
                        continue
                    one_hot = ','.join(['1' if rdict.get(cat, False) is True else '0' for cat in categories])
                all_list.append(f'{dirname}/{second_name}/{file}\t{one_hot}\n')

    with open(os.path.join(root_path, 'all_list.txt'), 'w', encoding='utf-8') as file:
        file.writelines(all_list)
    test_list = all_list[::split_ratio]
    with open(os.path.join(root_path, 'test.txt'), 'w', encoding='utf-8') as file:
        file.writelines(test_list)
    del all_list[::split_ratio]
    with open(os.path.join(root_path, 'train.txt'), 'w', encoding='utf-8') as file:
        file.writelines(all_list)


if __name__ == '__main__':
    process(os.getcwd(), 10)
    print('\nAll process success\n')
