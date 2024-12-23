import argparse
import itertools
import json
import os
import sys
from concurrent import futures

import boto3
import botocore
import numpy as np
import pandas as pd
import PIL.Image
import tqdm

try:
    import pycocotools.coco
except ImportError:
    sys.exit('Please install pycocotools:\n\n    pip install pycocotools\n')

# 生成的数据集允许的标签列表
# fmt: off
categories = [
    ['Bird',    'Bird',     '/m/015p6'],    # /m/015p6, Bird, 鸟
    ['Cat',     'Cat',      '/m/01yrx'],    # /m/01yrx, Cat, 猫
    ['Dog',     'Dog',      '/m/0bt9lr'],   # /m/0bt9lr, Dog, 狗
    ['Rabbit',  'Rabbit',   '/m/06mf6'],    # /m/06mf6, Rabbit, 兔子
    ['Mouse',   'Mouse',    '/m/04rmv'],    # /m/04rmv, Mouse, 老鼠
    ['Mouse',   'Hamster',  '/m/03qrc'],    # /m/03qrc, Hamster, 仓鼠
    ['Mouse',   'Squirrel', '/m/071qp'],    # /m/071qp, Squirrel, 松鼠
]
# fmt: on


# 检查 COCO 文件是否有问题
def checkCOCO(coco_file):
    coco_api = pycocotools.coco.COCO(coco_file)
    img_ids = sorted(list(coco_api.imgs.keys()))
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if 'minival' not in coco_file:
        ann_ids = [ann['id'] for anns_per_image in anns for ann in anns_per_image]
        if len(set(ann_ids)) != len(ann_ids):
            sys.exit(f"\n\n\033[1;31m Annotation ids in '{coco_file}' are not unique!\033[0m")


# 创建 coco
def create_COCO(args, task, csv_file, label_code):
    # 创建从 LabelName 映射到索引的字典
    label_set = {}
    label_dict = {}
    for dst, _, raw_name in categories:
        if dst not in label_set:
            label_set[dst] = len(label_set)
        label_dict[raw_name] = label_set[dst]
    # 提前获取所有图片尺寸信息并存储在字典中
    print('[info] Start obtaining image size information...')
    images = sorted([f.split('.')[0] for f in os.listdir(f'./{args.down_folder}/{task}/') if f.endswith('.jpg')])
    images_info = {}
    for image in tqdm.tqdm(images, leave=True, colour='CYAN'):
        image_path = f'{task}/{image}.jpg'
        width, height = PIL.Image.open(f'./{args.down_folder}/{image_path}').size
        assert width > 0 and height > 0, f'Invalid image: {image_path}'
        images_info[image] = (width, height, image_path)
    # 读取标签文件
    print('[info] Start preprocessing the label dictionary file...')
    annotations = pd.read_csv(csv_file, usecols=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])
    anns_grouped = annotations.groupby('ImageID')
    # 开始遍历图片列表
    print('[info] Start create coco file...')
    bbox_id = 0
    coco = dict(categories=[], images=[], annotations=[])  # 建立 coco json 格式
    for image_id, image in enumerate(tqdm.tqdm(images, leave=True, colour='CYAN')):
        anns = anns_grouped.get_group(image) if image in anns_grouped.groups else pd.DataFrame()
        anns = anns[anns['LabelName'].isin(label_code)]
        if anns.empty:
            continue
        width, height, path = images_info[image]
        coco['images'].append(dict(id=image_id, file_name=path, width=width, height=height))
        name_list = [label_dict[name] for name in anns['LabelName']]
        xmin_list = np.array(anns['XMin'].tolist()) * width
        xmax_list = np.array(anns['XMax'].tolist()) * width
        ymin_list = np.array(anns['YMin'].tolist()) * height
        ymax_list = np.array(anns['YMax'].tolist()) * height
        for i, (name, xmin, xmax, ymin, ymax) in enumerate(zip(name_list, xmin_list, xmax_list, ymin_list, ymax_list)):
            bbox = dict(
                id=bbox_id + i,
                image_id=image_id,
                category_id=name,
                bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                segmentation=[[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]],
                area=(xmax - xmin) * (ymax - ymin),
                iscrowd=0,
            )
            coco['annotations'].append(bbox)
        bbox_id += len(anns)
    # 导出到文件
    coco['categories'] = [{'id': id, 'name': cat, 'supercategory': cat} for cat, id in label_set.items()]
    with open(f'./{task}.json', 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=4)
    checkCOCO(f'./{task}.json')  # 检查COCO文件是否正确


def create_yolo(args, task, csv_file, label_code, anns_folder='yolo_anns'):
    # 创建从 LabelName 映射到索引的字典
    label_set = {}
    label_dict = {}
    for dst, _, raw_name in categories:
        if dst not in label_set:
            label_set[dst] = len(label_set)
        label_dict[raw_name] = label_set[dst]
    # 创建结果文件夹
    os.makedirs(f'./{anns_folder}/{task}', exist_ok=True)
    # 读取标签文件
    print('[info] Start preprocessing the label dictionary file...')
    annotations = pd.read_csv(csv_file, usecols=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])
    anns_grouped = annotations.groupby('ImageID')
    # 开始遍历图片列表
    image_bbox_dict = {}
    images = sorted([f.split('.')[0] for f in os.listdir(f'./{args.down_folder}/{task}/') if f.endswith('.jpg')])
    print('[info] Start create yolo annotation file...')
    for image in tqdm.tqdm(images, leave=True, colour='CYAN'):
        # 获取标签信息
        anns = anns_grouped.get_group(image) if image in anns_grouped.groups else pd.DataFrame()
        anns = anns[anns['LabelName'].isin(label_code)]
        if anns.empty:
            continue
        bbox = []
        for name, xmin, xmax, ymin, ymax in zip(
            anns['LabelName'], anns['XMin'], anns['XMax'], anns['YMin'], anns['YMax']
        ):
            width, height = xmax - xmin, ymax - ymin
            x_center, y_center = xmin + width / 2, ymin + height / 2
            bbox.append(f'{label_dict[name]} {x_center} {y_center} {width} {height}\n')
        image_bbox_dict[image] = bbox
    # 批量写入文件
    print('[info] Start writing yolo annotation file...')
    for image, bbox in tqdm.tqdm(image_bbox_dict.items(), leave=True, colour='CYAN'):
        with open(f'./{anns_folder}/{task}/{image}.txt', 'w', encoding='utf-8') as f:
            f.writelines(bbox)


# 获取标签列表
def get_label_list(args):
    classes_file = './csv_folder/class-descriptions-boxable.csv'
    assert os.path.isfile(classes_file), 'classes_file is not exits.'
    classes = pd.read_csv(classes_file, header=None)
    label_name = [c[1] for c in args.classes]
    label_code = [classes.loc[classes[1] == c[1]].values[0][0] for c in args.classes]
    assert len(label_name) == len(label_code)
    return label_name, label_code


# 获取图片列表
def get_images_list(args, csv_file, label_name, label_code, download_folder):
    # 第一次筛选图片列表 - 从 csv 文件
    annotations = pd.read_csv(csv_file, usecols=['ImageID', 'LabelName'])
    images_list = set()
    for i, code in enumerate(label_code):
        imgs = annotations['ImageID'][annotations.LabelName == code].values
        imgs = set(imgs)
        len_imgs = len(imgs)
        print(f'[info] Found {len_imgs} online images for {label_name[i]}.')
        if args.down_limit > 0 and len_imgs > args.down_limit:
            print(f'\tLimiting to {args.down_limit} images.')
            imgs = set(itertools.islice(imgs, args.down_limit))
        images_list.update(imgs)
    assert images_list
    # 第二次筛选图片列表 - 剔除已经下载的部分
    downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_folder) if f.endswith('.jpg')]
    return list(set(images_list) - set(downloaded_images_list))


# 下载单文件
def download_one_image(bucket, task, image_id, download_folder):
    try:
        bucket.download_file(f'{task}/{image_id}.jpg', os.path.join(download_folder, f'{image_id}.jpg'))
    except botocore.exceptions.ClientError as exception:
        sys.exit(f'ERROR when downloading image `{task}/{image_id}`: {str(exception)}')


# 多线程下载文件
def download_all_images(image_list, task, download_folder, num_processes):
    bucket = boto3.resource('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(
        'open-images-dataset'
    )
    progress_bar = tqdm.tqdm(total=len(image_list), desc='Downloading', leave=True)
    with futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        all_futures = [
            executor.submit(download_one_image, bucket, task, image_id, download_folder) for image_id in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()


# 解析传入参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='从 open-images-dataset 下载数据集并生成 COCO 标签文件.')
    parser.add_argument(
        '--tasks',
        choices=['train', 'test', 'validation', 'all'],
        default='validation',
        help="tasks choices: ['train', 'test', 'validation', 'all']",
    )
    parser.add_argument('--classes', type=str, nargs='+')
    parser.add_argument('--skip_coco', action='store_true', help='skip create coco file.')
    parser.add_argument('--skip_down', action='store_true', help='skip download.')
    parser.add_argument('--yolo', action='store_true', help='create yolo ann file.')
    parser.add_argument('--down_threads', type=int, default=16, help='Number of parallel processes to use.')
    parser.add_argument('--down_folder', type=str, default='dataset', help='Folder where to download the images.')
    parser.add_argument('--down_limit', type=int, default=0, help='Optional limit on number of images to download.')
    args = parser.parse_args()

    if args.skip_down and args.skip_coco:
        sys.exit('skip_down and skip_coco, do nothing.')

    # 初始化任务列表
    if args.tasks == 'all':
        args.tasks = ['train', 'test', 'validation']
    else:
        args.tasks = [args.tasks]

    # 初始化标签列表
    args.classes = [[c, c] for c in args.classes] if args.classes else categories

    return args


def main():
    args = parse_arguments()
    label_name, label_code = get_label_list(args)
    for task in args.tasks:
        print(f'\n[info] start download task: {task}')
        download_folder = os.path.join(os.getcwd(), args.down_folder, task)
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        # check annotations_file
        csv_file = f'./csv_folder/{task}-annotations-bbox.csv'
        assert os.path.isfile(csv_file), 'annotations_file is not exits.'
        # download
        if not args.skip_down:
            if images_list := get_images_list(args, csv_file, label_name, label_code, download_folder):
                print('[info] start download...')
                download_all_images(images_list, task, download_folder, args.down_threads)
                print('[info] Done!')
            else:
                print('[info] All images already downloaded.')
        # create coco
        if not args.skip_coco:
            if not args.yolo:
                assert not os.path.exists(f'./{task}.json'), '待创建的标签文件已存在!'
                create_COCO(args, task, csv_file, label_code)
            else:
                create_yolo(args, task, csv_file, label_code)
    print('\nAll process success.\n')


# 从 open_images_dataset 下载指定的类的图片
# https://storage.googleapis.com/openimages/web/download_v7.html#download-manually
# https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv
# https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
# https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv
# step1: 定义 categories 数组
# step2: 执行命令开始下载 python3 tools_bbox_oid2coco.py --tasks all
if __name__ == '__main__':
    main()
