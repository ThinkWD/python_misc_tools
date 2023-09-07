# -*- coding=utf-8 -*-

import os
import sys
import json
import tqdm
import boto3
import botocore
import argparse
import itertools
import imagesize
import pandas as pd
from concurrent import futures


try:
    import pycocotools.coco
except ImportError:
    sys.exit("Please install pycocotools:\n\n    pip install pycocotools\n")

# 生成的数据集允许的标签列表
# fmt: off
categories = [
	['Person', 				'Person', 				    ],
    ['Desk', 				'Desk', 				    ],
    ['Ladder', 				'Ladder', 				    ],
    ['Helmet', 				'Helmet', 				    ],
    ['Tin can', 			'Tin can', 				    ],
    ['Knife', 				'Knife', 				    ],
    ['Scissors', 			'Scissors',             	],
    ['Adhesive tape', 		'Adhesive tape', 		  	],
    ['Hammer',              'Hammer',               	],
    ['Tripod',              'Tripod',				  	],
    ['Ball', 				'Ball (Object)',			],
    ['Plastic bag', 		'Plastic bag',				],
    ['Luggage and bags', 	'Luggage and bags',			],
    ['Laptop',              'Laptop',			    	],
    ['Waste container',		'Waste container',			],

    ['Chair', 			    'Chair',					],
    ['Chair', 				'Stool',                    ],

    ['Box', 				'Box',                      ],
    ['Box', 				'Facial tissue holder',     ],

    ['Bottle', 				'Bottle', 				    ],
    ['Bottle', 				'Salt and pepper shakers',  ],
    
    ['Screwdriver', 		'Screwdriver',			    ],
    ['Screwdriver', 		'Chisel', 				    ],
    
    ['Drill', 				'Drill (Tool)',			    ],
    ['Drill', 				'Grinder', 				    ],
    
    ['Wrench',				'Wrench', 				    ],
    ['Wrench',				'Ratchet (Device)',		    ],
    
    ['Flashlight', 			'Flashlight', 			    ],
    ['Flashlight', 			'Torch', 				    ],
]
# fmt: on


# 检查 COCO 文件是否有问题
def checkCOCO(coco_file):
    coco_api = pycocotools.coco.COCO(coco_file)
    img_ids = sorted(list(coco_api.imgs.keys()))
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if "minival" not in coco_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        if len(set(ann_ids)) != len(ann_ids):
            sys.exit(f"\n\n\033[1;31m Annotation ids in '{coco_file}' are not unique!\033[0m")


# 创建 coco
def create_COCO(args, task, csv_file, label_code):
    label_set = {c[0] for c in args.classes}
    label_list = []
    for c in args.classes:
        if c[0] in label_set:
            label_list.append(c[0])
            label_set.remove(c[0])
    # 获取文件夹下所有图片名并排序
    images = sorted([f.split('.')[0] for f in os.listdir(f"./{args.down_folder}/{task}/") if f.endswith('.jpg')])
    # 读取标签文件
    annotations = pd.read_csv(csv_file, usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])
    # 开始遍历图片列表
    bbox_id = 0
    last_index = 0
    coco = dict(categories=[], images=[], annotations=[])  # 建立 coco json 格式
    for image_id, image in enumerate(tqdm.tqdm(images, desc=f"{task}", leave=True, colour="CYAN")):
        # 获取图片信息
        img_path = f"{task}/{image}.jpg"
        width, height = imagesize.get(f"./{args.down_folder}/{img_path}")
        assert width > 0 and height > 0
        # 获取标签信息
        anns = annotations.loc[annotations['ImageID'] == image]
        anns = anns[anns['LabelName'].isin(label_code)]
        if anns.empty:
            continue
        coco["images"].append(dict(id=image_id, file_name=img_path, width=width, height=height))
        category_ids = [label_list.index(args.classes[label_code.index(name)][0]) for name in anns['LabelName']]
        coco["annotations"].extend(
            [
                {
                    "id": bbox_id + i,
                    "image_id": image_id,
                    "category_id": category_ids[i],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "segmentation": [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0,
                }
                for i, (xmin, xmax, ymin, ymax) in enumerate(
                    zip(anns['XMin'] * width, anns['XMax'] * width, anns['YMin'] * height, anns['YMax'] * height)
                )
            ]
        )
        bbox_id += len(anns)
        end_index = anns.index[-1]
        annotations = annotations[end_index - last_index + 1 :]
        last_index = end_index + 1
    # 导出到文件
    coco["categories"] = [{"id": id, "name": cat, "supercategory": cat} for id, cat in enumerate(label_list)]
    with open(f"./{task}.json", "w", encoding='utf-8') as f:
        json.dump(coco, f, indent=4)
    checkCOCO(f"./{task}.json")  # 检查COCO文件是否正确


# 获取标签列表
def get_label_list(args):
    classes_file = "./csv_folder/class-descriptions-boxable.csv"
    assert os.path.isfile(classes_file), "classes_file is not exits."
    classes = pd.read_csv(classes_file, header=None)
    label_name = [c[1] for c in args.classes]
    label_code = [classes.loc[classes[1] == c[1]].values[0][0] for c in args.classes]
    assert len(label_name) == len(label_code)
    return label_name, label_code


# 获取图片列表
def get_images_list(args, csv_file, label_name, label_code, download_folder):
    # 第一次筛选图片列表 - 从 csv 文件
    annotations = pd.read_csv(csv_file, usecols=["ImageID", "LabelName"])
    images_list = set()
    for i, code in enumerate(label_code):
        imgs = annotations['ImageID'][annotations.LabelName == code].values
        imgs = set(imgs)
        len_imgs = len(imgs)
        print(f"[info] Found {len_imgs} online images for {label_name[i]}.")
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
        print(f"\n[info] start download task: {task}")
        download_folder = os.path.join(os.getcwd(), args.down_folder, task)
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        # check annotations_file
        csv_file = f"./csv_folder/{task}-annotations-bbox.csv"
        assert os.path.isfile(csv_file), "annotations_file is not exits."
        # download
        if not args.skip_down:
            if images_list := get_images_list(args, csv_file, label_name, label_code, download_folder):
                print("[info] start download...")
                download_all_images(images_list, task, download_folder, args.down_threads)
                print("[info] Done!")
            else:
                print('[info] All images already downloaded.')
        # create coco
        if not args.skip_coco:
            assert not os.path.exists(f"./{task}.json"), "待创建的标签文件已存在!"
            print("[info] start create coco file...")
            create_COCO(args, task, csv_file, label_code)
    print("\nAll process success.\n")


if __name__ == "__main__":
    main()
