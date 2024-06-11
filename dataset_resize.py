# -*- coding=utf-8 -*-

import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

##################################################################
#
#   数据集缩放工具
#   遍历文件夹, 将图片连同数据集标注的坐标整体进行缩放
#   支持目标检测 voc 格式和分割的 labelme 格式
#
##################################################################


# 取出 xml 内容 (length 预期长度, 为 0 则不检查)
def getXmlValue(root, name, length):
    XmlValue = root.findall(name)
    if length > 0:
        if len(XmlValue) != length:
            raise Exception("The size of %s is supposed to be %d, but is %d." % (name, length, len(XmlValue)))
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


def scale_img(imgpath, savepath, scalerate):
    raw_img = cv2.imread(imgpath)
    raw_height, raw_width = raw_img.shape[:2]
    res_height = int(raw_height * scalerate)
    res_width = int(raw_width * scalerate)
    resized_img = cv2.resize(raw_img, (res_width, res_height), interpolation=cv2.INTER_AREA)
    background = np.zeros(raw_img.shape, dtype=np.uint8)
    offset_x = random.randint(0, abs(raw_width - res_width))
    offset_y = random.randint(0, abs(raw_height - res_height))
    background[offset_y : offset_y + res_height, offset_x : offset_x + res_width] = resized_img
    cv2.imwrite(savepath, background)
    return offset_x, offset_y


def scale_det(xmlpath, savepath, scalerate, offset_x, offset_y):
    assert os.path.isfile(xmlpath)
    try:
        tree = ET.parse(xmlpath)  # 打开文件
        root = tree.getroot()  # 获取根节点
        for obj in getXmlValue(root, "object", 0):
            bndbox = getXmlValue(obj, "bndbox", 1)
            xmin = getXmlValue(bndbox, "xmin", 1)
            xmin.text = str(int(float(xmin.text) * scalerate + offset_x))
            ymin = getXmlValue(bndbox, "ymin", 1)
            ymin.text = str(int(float(ymin.text) * scalerate + offset_y))
            xmax = getXmlValue(bndbox, "xmax", 1)
            xmax.text = str(int(float(xmax.text) * scalerate + offset_x))
            ymax = getXmlValue(bndbox, "ymax", 1)
            ymax.text = str(int(float(ymax.text) * scalerate + offset_y))
        tree.write(savepath, encoding="UTF-8")
    except Exception as e:
        raise Exception(f"Failed to parse XML file: {xmlpath}, {e}")


def scale_seg(jsonpath, savepath, relative_path, scalerate, offset_x, offset_y):
    with open(jsonpath, "r", encoding="utf-8") as file:
        data = json.load(file)
    data["imageData"] = None
    data["imagePath"] = relative_path
    for shape in data["shapes"]:
        assert shape["shape_type"] == "polygon"
        for p in shape["points"]:
            p[0] = p[0] * scalerate + offset_x
            p[1] = p[1] * scalerate + offset_y
    with open(savepath, "w", encoding='utf-8') as file_out:
        json.dump(data, file_out, indent=4)


def scale(root_path, outdir, scalerate, name_offset):
    assert scalerate > 0 and scalerate < 1
    assert os.path.isdir(f"{root_path}/imgs"), "图片文件夹不存在!"
    assert not os.path.isdir(f"{root_path}/{outdir}"), "目标文件夹已存在!"
    os.makedirs(f"{root_path}/{outdir}/imgs")
    if os.path.isdir(f"{root_path}/anns"):
        os.makedirs(f"{root_path}/{outdir}/anns")
    if os.path.isdir(f"{root_path}/anns_seg"):
        os.makedirs(f"{root_path}/{outdir}/anns_seg")
    # get images list
    imgs_list = [f for f in os.listdir(f"{root_path}/imgs") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    namebit = 6 if len(imgs_list) > 9999 else 4
    # prosess
    for idx, file in enumerate(tqdm(imgs_list, leave=True, ncols=100, colour="CYAN")):
        raw_name, extension = os.path.splitext(os.path.basename(file))
        out_name = str(idx + name_offset).zfill(namebit)
        # scale img
        raw_img_path = f"{root_path}/imgs/{raw_name}{extension}"
        out_img_path = f"{root_path}/{outdir}/imgs/{out_name}{extension}"
        offset_x, offset_y = scale_img(raw_img_path, out_img_path, scalerate)
        # scale det ann
        raw_det_path = f"{root_path}/anns/{raw_name}.xml"
        if os.path.isfile(raw_det_path):
            out_det_path = f"{root_path}/{outdir}/anns/{out_name}.xml"
            scale_det(raw_det_path, out_det_path, scalerate, offset_x, offset_y)
        # scale seg ann
        raw_seg_path = f"{root_path}/anns_seg/{raw_name}.json"
        if os.path.isfile(raw_seg_path):
            out_seg_path = f"{root_path}/{outdir}/anns_seg/{out_name}.json"
            relative_path = f"../imgs/{out_name}{extension}"
            scale_seg(raw_seg_path, out_seg_path, relative_path, scalerate, offset_x, offset_y)


if __name__ == "__main__":
    scale(os.getcwd(), "sacle", 0.75, 100)
