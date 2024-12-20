import os

import numpy as np
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm

from module import find_dir, find_img, get_color_map, parse_labelme

##################################################################
#
#   此文件用于语义分割数据集转换格式, 从 labelme 多边形标注转为 VOC 格式, 并生成 mmseg 或 paddleseg 训练所需文件
#
#   COCO 格式用于 实例分割训练, VOC 格式用于 语义分割训练
#
##################################################################


def generate(img_path, seg_path, class_name_mapping):
    # check image
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size
    assert img_width > 0 and img_height > 0
    masks, _ = parse_labelme(seg_path, img_width, img_height)
    # generate label mask
    label_mask = np.zeros((img_height, img_width), dtype=np.int8)
    for instance, mask in masks.items():
        label_mask[mask] = class_name_mapping.get(instance[0], 0)
    return label_mask


def process(root_path, split_ratio, format='paddle'):
    # init class_names
    with open(f'{root_path}/classes.txt', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    assert len(class_names) < 255, f'check {root_path}/classes.txt'
    assert class_names[0] == '_background_', f'check {root_path}/classes.txt'
    class_name_to_id = {name: i for i, name in enumerate(class_names)}
    color_map = np.asarray(get_color_map(len(class_names))).flatten().tolist()

    # get path
    imgs_path = os.path.join(root_path, 'imgs')
    anns_path = os.path.join(root_path, 'anns_seg')
    pngs_path = os.path.join(root_path, 'anns_png')
    assert os.path.isdir(imgs_path), 'imgs directory not exists.'
    assert os.path.isdir(anns_path), 'anns_seg directory not exists.'

    # start work
    dataset = []
    for dir in find_dir(imgs_path):
        os.makedirs(os.path.join(pngs_path, dir), exist_ok=True)
        # 获取img文件列表
        imgs_dir_path = os.path.join(imgs_path, dir)
        if not os.path.isdir(imgs_dir_path):
            continue
        imgs_list = find_img(imgs_dir_path)
        # 遍历图片列表
        for file in tqdm(imgs_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN'):
            raw_name, extension = os.path.splitext(file)
            img_path = f'{imgs_path}/{dir}/{raw_name}{extension}'
            seg_path = f'{anns_path}/{dir}/{raw_name}.json'
            # get label mask
            mask = generate(img_path, seg_path, class_name_to_id)
            # save png image. (assume label ranges [0, 255] for uint8)
            if mask.min() < 0 or mask.max() > 255:
                raise Exception(f'[{seg_path}] Cannot save the pixel-wise class label as PNG.')
            lbl_pil = PIL.Image.fromarray(mask.astype(np.uint8), mode='P')
            lbl_pil.putpalette(color_map)
            lbl_pil.save(f'{pngs_path}/{dir}/{raw_name}.png')
            # set label string
            if format == 'paddle':
                dataset.append(f'imgs/{dir}/{raw_name}{extension} anns_png/{dir}/{raw_name}.png\n')
            elif format == 'mmlab':
                dataset.append(f'{dir}/{raw_name}\n')
            else:
                raise Exception('Only support Paddle OCR format and mmlab OCR format')

    with open(os.path.join(root_path, 'all_list.txt'), 'w', encoding='utf-8') as file:
        file.writelines(dataset)
    test_data = dataset[::split_ratio]
    with open(os.path.join(root_path, 'test.txt'), 'w', encoding='utf-8') as file:
        file.writelines(test_data)
    del dataset[::split_ratio]
    with open(os.path.join(root_path, 'train.txt'), 'w', encoding='utf-8') as file:
        file.writelines(dataset)


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == '__main__':
    process(os.getcwd(), 5)
