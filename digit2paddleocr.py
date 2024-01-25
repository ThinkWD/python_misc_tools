import os
import json
import PIL.Image
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 判断点 point 是否在矩形 rect 内部. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point(r, p):
    return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]


# 取出 xml 内容 (length 预期长度，为 0 则不检查)
def getXmlValue(root, name, length):
    # root为xml文件的根节点，name是子节点，作用为取出子节点内容
    XmlValue = root.findall(name)
    # 检查取出的值长度是否符合预期; 0 不检查
    if len(XmlValue) == 0:
        raise NotImplementedError(f"Can not find {name} in {root.tag}.")
    if length > 0:
        if len(XmlValue) != length:
            raise NotImplementedError("The size of %s is supposed to be %d, but is %d." % (name, length, len(XmlValue)))
        if length == 1:
            XmlValue = XmlValue[0]
    return XmlValue


# 解析单个 labelimg 标注文件(xml)
def parse_labelimg(xml_path, image_width, image_height):
    assert os.path.isfile(xml_path), f"标签文件不存在: {xml_path}"
    # 读标签文件
    try:
        tree = ET.parse(xml_path)  # 打开文件
        root = tree.getroot()  # 获取根节点
    except Exception:
        print(f"Failed to parse XML file: {xml_path}")
        exit()
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
    return bbox_dict


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_path, image_width, image_height):
    assert os.path.isfile(json_path), f"标签文件不存在: {json_path}"
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


# 单个图片
def generate(img_path, xml_path, json_path, save_dir, save_file, keep_ratio, format="paddle"):
    # check image
    assert os.path.isfile(img_path), f"图片文件不存在: {img_path}"
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelme anns file
    shape_labels, shape_points, shape_center = parse_labelme(json_path, width, height)
    # parse labelimg anns file
    label_string = []
    bbox_dict = parse_labelimg(xml_path, width, height)
    for idx, box in enumerate(bbox_dict):
        anns = []
        if keep_ratio:
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            img_length = max(box_width, box_height)
            offset = np.array([max((img_length - box_width) // 2, 0), max((img_length - box_height) // 2, 0)])
        # generate paddleocr label string
        box = np.array(box)  # 将矩形框转换为 numpy 数组
        for i, shape in enumerate(shape_points):
            if not rectangle_include_point(box, shape_center[i]):
                continue
            # 将多边形约束到矩形范围内
            new_shape = shape
            for p in new_shape:
                p[0] = max(box[0], min(p[0], box[2]))
                p[1] = max(box[1], min(p[1], box[3]))
            new_shape = new_shape - box[:2]
            if keep_ratio:
                new_shape = new_shape + offset

            if format == "paddle":
                ann = {"transcription": shape_labels[i], "points": new_shape.tolist()}
            elif format == "mmlab":
                ns_tl = new_shape.min(axis=0).tolist()
                ns_br = new_shape.max(axis=0).tolist()
                ann = {
                    "iscrowd": 0,
                    "category_id": 1,
                    "bbox": [ns_tl[0], ns_tl[1], ns_br[0] - ns_tl[0], ns_br[1] - ns_tl[1]],
                    "segmentation": new_shape.reshape(1, -1).tolist(),
                    "text": shape_labels[i],
                }
            else:
                print("Only support Paddle OCR format and mmlab OCR format")
                exit()

            anns.append(ann)
        # crop and save crop img
        if len(anns) > 0:
            path = f"{save_file}_{idx}.jpg"
            crop_img = img.crop(box).convert("RGB")
            if keep_ratio:
                res = PIL.Image.new("RGB", (img_length, img_length), (0, 0, 0))
                res.paste(crop_img, (offset[0], offset[1]))
                res.save(os.path.join(save_dir, path))
            else:
                crop_img.save(os.path.join(save_dir, path))

            if format == "paddle":
                result = f"{path}\t{json.dumps(anns, ensure_ascii=False)}\n"
            elif format == "mmlab":
                result = {"file_name": path, "height": height, "width": width, "annotations": anns}
                result = f"{json.dumps(result, ensure_ascii=False)}\n"
            else:
                print("Only support Paddle OCR format and mmlab OCR format")
                exit()

            label_string.append(result)
    return label_string


def process(root_path, save_dir, split, keep_ratio):
    work_path = os.path.join(root_path, "src")
    save_path = os.path.join(root_path, save_dir)
    assert os.path.isdir(work_path), f"数据集不存在: {work_path}"
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
            for file in tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
                # 获取文件名(带多文件夹的相对路径)
                raw_name, extension = os.path.splitext(file)
                img_path = f"src/{pre_dir}/imgs/{raw_name}{extension}"
                xml_path = f"src/{pre_dir}/anns/{raw_name}.xml"
                json_path = f"src/{pre_dir}/anns_seg/{raw_name}.json"
                img_save_path = f"dataset/{pre_dir}/{raw_name}"
                # 解析单个 ann 文件
                label_string = generate(img_path, xml_path, json_path, save_dir, img_save_path, keep_ratio)
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
