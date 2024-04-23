# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


##################################################################
#
#   此文件用于批量修改 VOC 数据集中的类名
#
#   change_list 中 key 为原类名, value 为目标类名
#   不需要修改的类也需要写入到 change_list 中, 设置 value 等于 key 即可
#
##################################################################


change_list = {"D000": "D000", "D001": "D001", "P000": "P000", "P001": "P001"}  # 指针仪表
# 数据集中出现的不在允许列表中的标签
skip_categories = []


# 取出 xml 内容 (length 预期长度, 为 0 则不检查)
def getXmlValue(root, name, length):
    XmlValue = root.findall(name)
    if length > 0:
        if len(XmlValue) != length:
            raise Exception("The size of %s is supposed to be %d, but is %d." % (name, length, len(XmlValue)))
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


if __name__ == "__main__":
    root_path = "D:\Work\Detector\__DataSet\数字仪表\step1_categories\src\P000"
    # 获取目录
    img_path = os.path.join(root_path, "imgs")
    assert os.path.isdir(img_path)
    ann_path = os.path.join(root_path, "anns")
    assert os.path.isdir(ann_path)
    bak_path = os.path.join(root_path, "anns_bak")
    assert not os.path.isdir(bak_path)
    os.rename(ann_path, bak_path)
    os.makedirs(ann_path)
    # 遍历img文件列表
    img_list = os.listdir(img_path)
    for file in tqdm(img_list, leave=True, ncols=100, colour="CYAN"):
        # 文件路径操作
        file = os.path.basename(file)
        filename = os.path.splitext(file)[0]  # 获取文件名不带后缀
        xmlpath = f"{bak_path}/{filename}.xml"  # 旧路径
        if not os.path.isfile(xmlpath):
            print(f"[error] 标签文件不存在：{xmlpath}")
            continue
        resxmlpath = f"{ann_path}/{filename}.xml"  # 新路径
        # 读取并修改
        check_label = True
        tree = ET.parse(xmlpath)  # 打开文件
        root = tree.getroot()  # 获取根节点
        # path = getXmlValue(root, "path", 1)
        # path.text = f"../imgs/{file}"
        for obj in getXmlValue(root, "object", 0):
            name = getXmlValue(obj, "name", 1)  # 取出 name 节点
            if name.text in change_list:
                name.text = change_list[name.text]
            else:
                check_label = False
        if check_label:
            tree.write(resxmlpath)  # 写入保存文件
        else:
            print(f"[w] 标签文件有问题：{xmlpath}")

    print("\nAll process success\n")
