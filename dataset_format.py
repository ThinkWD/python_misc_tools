import json
import os
import shutil
import xml.etree.ElementTree as ET

import PIL.Image
from tqdm import tqdm

##################################################################
#
#   此文件用于数据集格式化, 操作: 重命名, 统一分辨率, 移除 labelme 标注中的图片数据
#
#   ./imgs      --> ./format/imgs
#   ./anns      --> ./format/anns
#   ./anns_seg  --> ./format/anns_seg
#
##################################################################

# 是否启用重命名图片及标签文件功能
enable_rename = True
name_offset = 0

# 是否启用 resize 图片及标签文件功能
enable_resize = True
target_size = (0, 0)


# 取出 xml 内容 (length 预期长度, 为 0 则不检查)
def getXmlValue(root, name, length):
    XmlValue = root.findall(name)
    if length > 0:
        if len(XmlValue) != length:
            raise Exception(f'The size of {name} is supposed to be {length}, but is {len(XmlValue)}.')
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


def pipeline_det(ann_path, dst_path, raw_size, target_size):
    if not os.path.isfile(ann_path):
        return
    if target_size[0] == 0 or target_size[1] == 0 or raw_size == target_size:
        shutil.copy(ann_path, dst_path)
        return
    try:
        tree = ET.parse(ann_path)  # 打开文件
        root = tree.getroot()  # 获取根节点
        # size
        size = getXmlValue(root, 'size', 1)
        width = getXmlValue(size, 'width', 1)
        width.text = str(target_size[0])
        height = getXmlValue(size, 'height', 1)
        height.text = str(target_size[1])
        # object
        x_scale = target_size[0] / raw_size[0]
        y_scale = target_size[1] / raw_size[1]
        for obj in getXmlValue(root, 'object', 0):
            bndbox = getXmlValue(obj, 'bndbox', 1)
            xmin = getXmlValue(bndbox, 'xmin', 1)
            xmin.text = str(round(float(xmin.text) * x_scale))
            ymin = getXmlValue(bndbox, 'ymin', 1)
            ymin.text = str(round(float(ymin.text) * y_scale))
            xmax = getXmlValue(bndbox, 'xmax', 1)
            xmax.text = str(round(float(xmax.text) * x_scale))
            ymax = getXmlValue(bndbox, 'ymax', 1)
            ymax.text = str(round(float(ymax.text) * y_scale))
        tree.write(dst_path, encoding='UTF-8')
    except Exception as e:
        raise Exception(f'Failed to parse XML file: {ann_path}, {e}')


def pipeline_seg(ann_path, dst_path, relative_path, raw_size, target_size):
    if not os.path.isfile(ann_path):
        return
    with open(ann_path, encoding='utf-8') as file_in:
        data = json.load(file_in)
    data['imageData'] = None
    data['imagePath'] = relative_path
    if target_size[0] > 0 and target_size[1] > 0 and raw_size != target_size:
        data['imageWidth'] = target_size[0]
        data['imageHeight'] = target_size[1]
        x_scale = target_size[0] / raw_size[0]
        y_scale = target_size[1] / raw_size[1]
        for shape in data['shapes']:
            for p in shape['points']:
                p[0] = p[0] * x_scale
                p[1] = p[1] * y_scale
    with open(dst_path, 'w', encoding='utf-8') as file_out:
        json.dump(data, file_out, indent=4)


def pipeline_image(src_path, dst_path, target_size):
    image = PIL.Image.open(src_path)
    image_size = image.size
    if target_size[0] > 0 and target_size[1] > 0 and image_size != target_size:
        image = image.resize(target_size, PIL.Image.BICUBIC)
        image.save(dst_path)
    else:
        shutil.copy(src_path, dst_path)
    return image_size


def pipeline(root_path):
    # dynamic set imgdir
    dirlist = [it.name for it in os.scandir(root_path) if it.is_dir() and it.name.startswith('imgs_lv')]
    imgdir = 'imgs' if len(dirlist) == 0 else dirlist[0]
    print(f'\n >> Images directory: {imgdir}')
    # check path
    assert os.path.isdir(f'{root_path}/{imgdir}'), '图片文件夹不存在!'
    assert not os.path.isdir(f'{root_path}/format'), '目标文件夹已存在!'
    os.makedirs(f'{root_path}/format/imgs')
    if os.path.isdir(f'{root_path}/anns'):
        os.makedirs(f'{root_path}/format/anns')
    if os.path.isdir(f'{root_path}/anns_seg'):
        os.makedirs(f'{root_path}/format/anns_seg')
    # get images list
    imgs_list = [f for f in os.listdir(f'{root_path}/{imgdir}') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(imgs_list) > 0, '图片文件夹下没有图片'
    # get resize target size
    resize_target_size = (0, 0)
    if enable_resize:
        if target_size[0] > 0 and target_size[1] > 0:
            resize_target_size = target_size
        else:
            first_image = PIL.Image.open(f'{root_path}/{imgdir}/{imgs_list[0]}')
            resize_target_size = first_image.size
        print(f'\n >> resize_target_size: {resize_target_size}\n')
    # prosess
    resize_count = 0
    namebit = 6 if len(imgs_list) + name_offset > 9999 else 4
    for idx, file in enumerate(tqdm(imgs_list, leave=True, ncols=100, colour='CYAN')):
        raw_name, extension = os.path.splitext(file)
        out_name = str(idx + name_offset).zfill(namebit) if enable_rename else raw_name
        # image
        img_src = f'{root_path}/{imgdir}/{file}'
        img_dst = f'{root_path}/format/imgs/{out_name}{extension}'
        image_size = pipeline_image(img_src, img_dst, resize_target_size)
        resize_count += image_size != resize_target_size
        # ann det
        det_src = f'{root_path}/anns/{raw_name}.xml'
        det_dst = f'{root_path}/format/anns/{out_name}.xml'
        pipeline_det(det_src, det_dst, image_size, resize_target_size)
        # ann seg
        seg_src = f'{root_path}/anns_seg/{raw_name}.json'
        seg_dst = f'{root_path}/format/anns_seg/{out_name}.json'
        seg_relative = f'../imgs/{out_name}{extension}'
        pipeline_seg(seg_src, seg_dst, seg_relative, image_size, resize_target_size)

    print(f'\n\033[1;33m[Warning] 被缩放的图片数量: \033[0m{resize_count}')


if __name__ == '__main__':
    pipeline(os.getcwd())
    print('\nAll process success\n')
