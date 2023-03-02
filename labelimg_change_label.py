# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


change_list = {"0000": "D000", "0010": "D001"}  # 数字仪表
# change_list = {"0000": "P000", "0001": "P001"}  # 指针仪表


# 取出 xml 内容 (length 预期长度，为 0 则不检查)
def getXmlValue(root, name, length):
    # root为xml文件的根节点，name是子节点，作用为取出子节点内容
    XmlValue = root.findall(name)
    # 检查取出的值长度是否符合预期; 0 不检查
    if len(XmlValue) == 0:
        raise NotImplementedError(f"Can not find {name} in {root.tag}.")
    if length > 0:
        if len(XmlValue) != length:
            raise NotImplementedError(
                "The size of %s is supposed to be %d, but is %d."
                % (name, length, len(XmlValue))
            )
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


if __name__ == "__main__":
    # 获取目录
    ann_path = os.path.join(os.getcwd(), "anns")
    assert os.path.exists(ann_path) and os.path.isdir(ann_path)
    ann_list = os.listdir(ann_path)
    # 设置 tqdm 进度条
    with tqdm(
        total=len(ann_list),  # 迭代总数
        leave=True,  # 进度条走完是否保留
        ncols=100,  # 进度条长度
        colour="CYAN",  # 颜色(BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)
    ) as pbar:
        # 遍历XML列表
        for file in ann_list:
            # 获取文件名(带多文件夹的相对路径)
            file = file.strip()
            xmlpath = f"{ann_path}/{file}"
            # 检查文件是否存在
            if not os.path.isfile(xmlpath):
                continue
            tree = ET.parse(xmlpath)  # 打开文件
            root = tree.getroot()  # 获取根节点
            for obj in getXmlValue(root, "object", 0):
                name = getXmlValue(obj, "name", 1)  # 取出 name 节点
                if name.text in change_list:
                    name.text = change_list[name.text]
            tree.write(xmlpath)  # 写入保存文件
            # 更新进度条
            pbar.update(1)

    print("\nAll process success\n")
