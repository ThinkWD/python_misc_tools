import os
import shutil
import stat

import imagesize
from lxml import etree, objectify
from pycocotools.coco import COCO
from tqdm import tqdm

##################################################################
#
#   此文件用于目标检测数据集转换格式, 从 COCO 格式转为 VOC 格式
#
##################################################################


def rm_read_only(tmp):
    if os.path.isfile(tmp):
        os.chmod(tmp, stat.S_IWRITE)
        os.remove(tmp)
    elif os.path.isdir(tmp):
        os.chmod(tmp, stat.S_IWRITE)
        shutil.rmtree(tmp)


def coco2voc(annfile, outdir):
    # 创建文件夹
    assert not os.path.exists(outdir)
    img_savepath = os.path.join(outdir, 'imgs')
    ann_savepath = os.path.join(outdir, 'anns')
    noann_dir = os.path.join(outdir, 'noann_result')
    notRGB_dir = os.path.join(outdir, 'notrgb_result')
    for dir in [outdir, img_savepath, ann_savepath, noann_dir, notRGB_dir]:
        os.makedirs(dir)

    # 加载 COCO 文件
    noann_cnt = 0  # 没有标注数据的图片计数
    notRGB_cnt = 0  # 非三通道RGB的图片计数
    coco = COCO(annfile)
    classes = {cat['id']: cat['name'] for cat in coco.dataset['categories']}
    imgIds = coco.getImgIds()
    for imgId in tqdm(imgIds):
        # 获取 img 信息
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        # 没有 ann
        if not len(anns):
            noann_cnt += 1
            shutil.copy(filename, noann_dir)
            continue
        else:
            shutil.copy(filename, img_savepath)

        # 解析 ann
        objs = []
        for ann in anns:
            name = classes[ann['category_id']]
            if 'bbox' in ann:
                bbox = ann['bbox']
                if int(bbox[2]) == 0 or int(bbox[3]) == 0:
                    continue
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [name, xmin, ymin, xmax, ymax]
                objs.append(obj)

        pos = filename.rfind('/')
        if pos > 0:
            xmlsavepath = f"{filename[pos + 1:filename.rindex('.')]}.xml"
        else:
            xmlsavepath = f"{filename[:filename.rindex('.')]}.xml"
        annopath = os.path.join(ann_savepath, xmlsavepath)
        width, height = imagesize.get(filename)
        # 创建xml
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('imgs'),
            E.filename(filename),
            E.source(E.database('COCO'), E.annotation('VOC'), E.image('COCO')),
            E.size(E.width(width), E.height(height), E.depth(3)),
            E.segmented(0),
        )
        for obj in objs:
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(
                E.name(obj[0]),
                E.pose('Unspecified'),
                E.truncated('0'),
                E.difficult(0),
                E.bndbox(E.xmin(obj[1]), E.ymin(obj[2]), E.xmax(obj[3]), E.ymax(obj[4])),
            )
            anno_tree.append(anno_tree2)
        etree.ElementTree(anno_tree).write(annopath, encoding='UTF-8', pretty_print=True)

    # 转换完成
    print('\n[Info] All process success\n')
    if noann_cnt > 0:
        print(f'[Info] {noann_cnt} 张图片没有instance标注信息, 已存放至 {noann_dir}')
    else:
        rm_read_only(noann_dir)
    if notRGB_cnt > 0:
        print(f'[Info] {notRGB_cnt} 张图片是非RGB图像, 已存放至 {notRGB_dir}')
    else:
        rm_read_only(notRGB_dir)


if __name__ == '__main__':
    coco2voc('./instance_test.json', './output')
