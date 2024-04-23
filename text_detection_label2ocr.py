import os
import json
import PIL.Image
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

##################################################################
#
#   此文件用于文本检测数据集转换格式
#
#   1. 从 labelimg 矩形标注裁出小图, 缩放后(可选), 保存到文件
#   2. 从 labelme 多边形标注提取文本框在小图上的相对坐标
#   3. 转为 mmocr 或 paddocr 训练所需的格式.
#
##################################################################


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 判断点 point 是否在矩形 rect 内部. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point(r, p):
    return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]


# 取出 xml 内容 (length 预期长度, 为 0 则不检查)
def getXmlValue(root, name, length):
    XmlValue = root.findall(name)
    if length > 0:
        if len(XmlValue) != length:
            raise Exception("The size of %s is supposed to be %d, but is %d." % (name, length, len(XmlValue)))
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


# 解析单个 labelimg 标注文件(xml)
def parse_labelimg(xml_path, image_width, image_height):
    if not os.path.isfile(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)  # 打开文件
        root = tree.getroot()  # 获取根节点
        # 验证图片与标签是否对应
        imgsize = getXmlValue(root, "size", 1)
        assert image_width == int(getXmlValue(imgsize, "width", 1).text), f"图片与标签不对应: {xml_path}"
        assert image_height == int(getXmlValue(imgsize, "height", 1).text), f"图片与标签不对应: {xml_path}"
        # 提取框信息
        bbox_dict = []
        for obj in getXmlValue(root, "object", 0):
            bndbox = getXmlValue(obj, "bndbox", 1)
            xmin = int(float(getXmlValue(bndbox, "xmin", 1).text) - 1)
            ymin = int(float(getXmlValue(bndbox, "ymin", 1).text) - 1)
            xmax = int(float(getXmlValue(bndbox, "xmax", 1).text))
            ymax = int(float(getXmlValue(bndbox, "ymax", 1).text))
            assert xmax > xmin and ymax > ymin and xmax <= image_width and ymax <= image_height, f"{xml_path}"
            bbox_dict.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        raise Exception(f"Failed to parse XML file: {xml_path}, {e}")
    return bbox_dict


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_path, image_width, image_height):
    if not os.path.isfile(json_path):
        return [], [], []
    # 处理标注文件中的多余数据
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if data["imageData"] != None:
        data["imageData"] = None
        image_name = os.path.basename(data["imagePath"])
        data["imagePath"] = f"../imgs/{image_name}"
        with open(json_path, "w", encoding="utf-8") as file_out:
            file_out.write(json.dumps(data))
    # 验证图片与标签是否对应
    assert image_width == int(data["imageWidth"]), f"图片与标签不对应: {json_path}"
    assert image_height == int(data["imageHeight"]), f"图片与标签不对应: {json_path}"
    # 遍历 shapes
    shape_labels = []
    shape_points = []
    shape_center = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            # parse point
            points = []
            for point in shape["points"]:
                points.extend([int(x) for x in point])
            assert len(points) >= 8 and len(points) % 2 == 0, f"Invalid polygon: {shape['points']}."
            point = np.array(points).reshape(-1, 2).astype(int)  # 将列表转为 numpy 矩阵
            # append
            shape_labels.append(shape["label"])
            shape_points.append(point)
            shape_center.append(point.mean(axis=0))

    return shape_labels, shape_points, shape_center


def generate_labelme_check_file(shapes, width, height, save_path, save_name):
    check_json = dict(
        version='5.1.1',
        flags={},
        shapes=[],
        imagePath=f'./{save_name}.jpg',
        imageData=None,
        imageHeight=height,
        imageWidth=width,
    )
    for shape in shapes:
        element = dict(
            label='Null',
            points=shape.tolist(),
            group_id=None,
            shape_type='polygon',
            flags={},
        )
        check_json['shapes'].append(element)
    with open(f"{save_path}/{save_name}.json", "w") as f:
        json.dump(check_json, f, indent=4)


def generate_format_label_string(labels, shapes, width, height, relative_path, format="paddle"):
    if format == "paddle":
        anns = []
        for i, shape in enumerate(shapes):
            anns.append({"transcription": labels[i], "points": shape.tolist()})
        result = f"{relative_path}\t{json.dumps(anns, ensure_ascii=False)}\n"

    elif format == "mmlab":
        anns = []
        for i, shape in enumerate(shapes):
            ns_tl = shape.min(axis=0).tolist()
            ns_br = shape.max(axis=0).tolist()
            ann = {
                "iscrowd": 0,
                "category_id": 1,
                "bbox": [ns_tl[0], ns_tl[1], ns_br[0] - ns_tl[0], ns_br[1] - ns_tl[1]],
                "segmentation": shape.reshape(1, -1).tolist(),
                "text": labels[i],
            }
            anns.append(ann)
        result = {"file_name": relative_path, "height": height, "width": width, "annotations": anns}
        result = f"{json.dumps(result, ensure_ascii=False)}\n"

    else:
        raise Exception("Only support Paddle OCR format and mmlab OCR format")

    return result


# 单个图片
def generate(img_path, xml_path, json_path, keep_ratio, save_root, save_relative, save_name, resize=736):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size
    assert img_width > 0 and img_height > 0
    # parse labelme anns file
    shape_labels, shape_points, shape_center = parse_labelme(json_path, img_width, img_height)
    # parse labelimg anns file
    bbox_dict = parse_labelimg(xml_path, img_width, img_height)
    # Start loop
    label_string = []
    save_path = os.path.join(save_root, save_relative)
    for idx, box in enumerate(bbox_dict):
        # get shapes
        box = np.array(box)  # 将矩形框转换为 numpy 数组
        shapes = []
        for i, shape in enumerate(shape_points):
            if not rectangle_include_point(box, shape_center[i]):
                continue
            new_shape = shape  # 将多边形约束到矩形范围内
            for p in new_shape:
                p[0] = max(box[0], min(p[0], box[2]))
                p[1] = max(box[1], min(p[1], box[3]))
            shapes.append(new_shape - box[:2])
        if len(shapes) == 0:
            continue

        # organize path
        raw_name = f"{save_name}_{idx}"
        rel_path = f"{save_relative}/{raw_name}.jpg"  # relative_path

        # crop and save crop img
        box_width = int(box[2] - box[0])
        box_height = int(box[3] - box[1])
        crop_img = img.crop(box).convert("RGB")
        if keep_ratio:
            # update params
            img_length = max(box_width, box_height)
            offset = np.array([max((img_length - box_width) // 2, 0), max((img_length - box_height) // 2, 0)])
            scale = resize / img_length if resize > img_length else 1
            box_width = resize if resize > img_length else img_length
            box_height = resize if resize > img_length else img_length
            for i in range(len(shapes)):
                shapes[i] = (shapes[i] + offset) * scale
            # pad and resize image
            res = PIL.Image.new("RGB", (img_length, img_length), (0, 0, 0))
            res.paste(crop_img, (offset[0], offset[1]))
            crop_img = res.resize((box_width, box_height), PIL.Image.BILINEAR)
        crop_img.save(f"{save_path}/{raw_name}.jpg")

        # generate anns label string
        label_string.append(generate_format_label_string(shape_labels, shapes, box_width, box_height, rel_path))

        # generate labelme check file
        generate_labelme_check_file(shapes, box_width, box_height, save_path, raw_name)

    return label_string


def process(root_path, save_dir, split, keep_ratio):
    work_path = os.path.join(root_path, "src")
    save_path = os.path.join(root_path, save_dir)
    assert os.path.isdir(work_path), f"数据集不存在: {work_path}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print(f"\n[info] start task...")
    with open(f"{save_path}/all_list.txt", "w", encoding='utf-8') as ann_file:
        # 遍历 root_path 下的子文件夹
        dirs = find_dir(work_path)
        for dir in dirs:
            # 获取子文件夹名
            pre_dir = os.path.basename(dir)
            save_sub_path = f"{save_path}/dataset/{pre_dir}"
            assert not os.path.isdir(save_sub_path), f"结果文件夹已经存在: {save_sub_path}"
            os.makedirs(save_sub_path)
            # 获取img文件列表
            img_path = os.path.join(dir, "imgs")
            assert os.path.isdir(img_path), f"图片文件夹不存在: {img_path}"
            img_list = [f for f in os.listdir(img_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            # 遍历 ann 文件列表
            save_relative = f"dataset/{pre_dir}"
            for file in tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
                # 获取文件名(带多文件夹的相对路径)
                raw_name, extension = os.path.splitext(file)
                img_path = f"{work_path}/{pre_dir}/imgs/{raw_name}{extension}"
                xml_path = f"{work_path}/{pre_dir}/anns/{raw_name}.xml"
                json_path = f"{work_path}/{pre_dir}/anns_seg/{raw_name}.json"
                # 解析单个 ann 文件
                label_string = generate(img_path, xml_path, json_path, keep_ratio, save_path, save_relative, raw_name)
                for str in label_string:
                    ann_file.write(str)
    with open(f"{save_path}/all_list.txt", "r", encoding='utf-8') as f:
        list_train = f.readlines()
    list_test = list_train[::split]
    with open(f"{save_path}/test.txt", "w", encoding='utf-8') as file:
        file.writelines(list_test)
    del list_train[::split]
    with open(f"{save_path}/train.txt", "w", encoding='utf-8') as file:
        file.writelines(list_train)


if __name__ == "__main__":
    process(os.getcwd(), "dataset", 10, True)
    print("\nAll process success\n")