import json
import os

import numpy as np
import PIL.Image
import pycocotools.mask
from tqdm import tqdm

from module import (
    checkCOCO,
    find_dir,
    find_img,
    get_color_map,
    get_matching_pairs,
    parse_labelimg,
    parse_labelme,
    shape_to_mask,
)

##################################################################
#
#   此文件用于 旋转框 第二步数据集转换格式, 从 labelme 多边形标注转为 COCO 格式
#
##################################################################


categories = ['switch']
skip_cates = []
generate_anns_check_image = False
palette = get_color_map(80)


def generate_labelme_check_file(shape, width, height, save_path, save_name):
    check_json = dict(
        version='2.3.6',
        flags={},
        shapes=[],
        imagePath=f'{save_name}.jpg',
        imageData=None,
        imageHeight=height,
        imageWidth=width,
    )
    element = dict(
        label='Null',
        points=shape.tolist(),
        group_id=None,
        description='',
        difficult=False,
        shape_type='polygon',
        flags={},
        attributes={},
    )
    check_json['shapes'].append(element)
    with open(f'{save_path}/{save_name}.json', 'w', encoding='utf-8') as f:
        json.dump(check_json, f, indent=4)


# 单个图片
def generate(img_path, det_path, seg_path, keep_ratio, save_root, save_relative, resize=512):
    # check image
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size
    assert img_width > 0 and img_height > 0
    # parse anns file
    bbox = parse_labelimg(det_path, img_width, img_height)
    masks, shapes = parse_labelme(seg_path, img_width, img_height)
    skip_cates.extend([id[0] for id, _ in shapes.items() if id[0] not in categories and id[0] not in skip_cates])
    shapes = {instance: shape for instance, shape in shapes.items() if instance[0] in categories}
    # get box shapes pairs
    pairs = get_matching_pairs(seg_path, bbox, shapes)
    if len(pairs) == 0:
        return []
    # generate anns
    imgs_dict = []
    anns_dict = []
    for idx, (box_instance, shape_instances) in enumerate(pairs.items()):
        box = np.array(bbox[box_instance])
        in_shapes = {instance: np.asarray(shapes[instance]).reshape(-1, 2) - box[:2] for instance in shape_instances}
        # crop and save crop img
        box_width = int(box[2] - box[0])
        box_height = int(box[3] - box[1])
        img_length = max(box_width, box_height)
        crop_img = img.crop(box).convert('RGB')
        if keep_ratio:
            # update image
            offset = np.array([max((img_length - box_width) // 2, 0), max((img_length - box_height) // 2, 0)])
            temp = PIL.Image.new('RGB', (img_length, img_length), (0, 0, 0))
            temp.paste(crop_img, (offset[0], offset[1]))
            # set resize
            scale = resize / img_length if resize > img_length else 1
            img_length = max(resize, img_length)
            crop_img = temp.resize((img_length, img_length), PIL.Image.BICUBIC)
            # update in_shapes and masks
            for instance, shape in in_shapes.items():
                in_shapes[instance] = (shape + offset) * scale
                masks[instance] = shape_to_mask([img_length, img_length], in_shapes[instance], 'rotation')
        rel_path = f'dataset/{save_relative}_{idx}.jpg'
        crop_img.save(f'{save_root}/{rel_path}')
        # generate coco label
        anns = []
        for instance, shape in in_shapes.items():
            generate_labelme_check_file(
                shape, img_length, img_length, f'{save_root}/dataset_viz', f'{save_relative}_{idx}'
            )
            shape = np.rint(shape).astype(int).reshape(1, -1).tolist()  # round to int
            label_id = categories.index(instance[0])
            t_mask = masks[instance]
            t_mask = np.asfortranarray(t_mask.astype(np.uint8))
            t_mask = pycocotools.mask.encode(t_mask)
            t_area = float(pycocotools.mask.area(t_mask))
            t_bbox = pycocotools.mask.toBbox(t_mask).flatten().tolist()
            ann = dict(id=0, image_id=0, category_id=label_id, bbox=t_bbox, segmentation=shape, area=t_area, iscrowd=0)
            anns.append(ann)
        imgs_dict.append(dict(id=0, file_name=rel_path, width=img_length, height=img_length))
        anns_dict.append(anns)
        # generate anns check image
        if not generate_anns_check_image:
            continue
        for index, (instance, shape) in enumerate(in_shapes.items()):
            # set mask to image
            mask = masks[instance].astype(np.uint8)
            mask[mask == 0] = 255
            mask[mask == 1] = 128  # 透明度 50 %
            mask = PIL.Image.fromarray(mask, mode='L')
            # make mask image
            color_img = PIL.Image.new('RGB', (img_length, img_length), palette[index % 80])
            crop_img = PIL.Image.composite(crop_img, color_img, mask)
        crop_img.save(f'{save_root}/dataset_viz/{save_relative}_{idx}.jpg')
    return imgs_dict, anns_dict


def process(root_path, save_dir, split, keep_ratio, all_reserve=0):
    print('\n[info] start task...')
    work_path = os.path.join(root_path, 'src')
    save_path = os.path.join(root_path, save_dir)
    assert os.path.isdir(work_path), f'数据集不存在: {work_path}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = 0
    train_bbox_id = 0
    test_img_id = 0
    test_bbox_id = 0
    anns_count = 0
    for dir in find_dir(work_path):
        # makedirs
        os.makedirs(os.path.join(save_path, 'dataset', dir), exist_ok=True)
        if generate_anns_check_image:
            os.makedirs(os.path.join(save_path, 'dataset_viz', dir), exist_ok=True)
        # 获取img文件列表
        imgs_dir_path = os.path.join(work_path, dir, 'imgs')
        if not os.path.isdir(imgs_dir_path):
            continue
        img_list = find_img(imgs_dir_path)
        all_reserve_dir = len(img_list) < all_reserve
        for file in tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN'):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f'{work_path}/{dir}/imgs/{raw_name}{extension}'
            det_path = f'{work_path}/{dir}/anns/{raw_name}.xml'
            seg_path = f'{work_path}/{dir}/anns_seg/{raw_name}.json'
            # get dict
            imgs_dict, anns_dict = generate(img_path, det_path, seg_path, keep_ratio, save_path, f'{dir}/{raw_name}')
            for i in range(len(imgs_dict)):
                anns_size = len(anns_dict[i])
                if anns_size == 0:
                    continue
                anns_count += 1
                # train dataset
                if all_reserve_dir or split <= 0 or anns_count % split != 0:
                    imgs_dict[i]['id'] = train_img_id
                    data_train['images'].append(imgs_dict[i].copy())
                    for idx, ann in enumerate(anns_dict[i]):
                        ann['image_id'] = train_img_id
                        ann['id'] = train_bbox_id + idx
                        data_train['annotations'].append(ann.copy())
                    train_img_id += 1
                    train_bbox_id += anns_size
                # test dataset
                if all_reserve_dir or split <= 0 or anns_count % split == 0:
                    imgs_dict[i]['id'] = test_img_id
                    data_test['images'].append(imgs_dict[i].copy())
                    for idx, ann in enumerate(anns_dict[i]):
                        ann['image_id'] = test_img_id
                        ann['id'] = test_bbox_id + idx
                        data_test['annotations'].append(ann.copy())
                    test_img_id += 1
                    test_bbox_id += anns_size
    print(f'\n训练集图片总数: {train_img_id}, 标注总数: {train_bbox_id}\n')
    print(f'测试集图片总数: {test_img_id}, 标注总数: {test_bbox_id}\n')
    # export to file
    for id, category in enumerate(categories):
        cat = {'id': id, 'name': category, 'supercategory': category}
        data_train['categories'].append(cat)
        data_test['categories'].append(cat)
    with open(f'{save_path}/train.json', 'w', encoding='utf-8') as f:
        json.dump(data_train, f, indent=4)
    checkCOCO(f'{save_path}/train.json')
    with open(f'{save_path}/test.json', 'w', encoding='utf-8') as f:
        json.dump(data_test, f, indent=4)
    checkCOCO(f'{save_path}/test.json')


if __name__ == '__main__':
    process(os.getcwd(), 'dataset_seg', 5, True)
    print('\nAll process success\n')
