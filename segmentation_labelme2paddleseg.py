import os
import json
import math
import numpy as np
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm


##################################################################
#
#   此文件用于语义分割数据集转换格式, 从 labelme 多边形标注转为 VOC 格式, 并生成 paddleseg 训练所需文件
#
#   COCO 格式用于 实例分割训练, VOC 格式用于 语义分割训练
#
##################################################################


def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
            color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
            color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    return color_map


# 保存 labelme 支持的形状类型
labelme_shape_type = ["circle", "rectangle", "line", "linestrip", "point", "polygon"]


# shape_to_mask
def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shape2label(img_size, shapes, class_name_mapping):
    label = np.zeros(img_size[:2], dtype=np.int32)
    for shape in shapes:
        shape_type = shape.get('shape_type', None)
        assert shape_type in labelme_shape_type, f"不支持的形状: {shape_type}"
        points = shape['points']
        class_name = shape['label']
        class_id = class_name_mapping[class_name]
        label_mask = shape_to_mask(img_size[:2], points, shape_type)
        label[label_mask] = class_id
    return label


def main(root_path, split_ratio):
    # init class_names
    class_names = ['__ignore__', '_background_', 'belt_L', 'belt_R', 'roller_L', 'roller_R']  # 0: 刻度, 1: 指针
    class_name_to_id = {name: i - 1 for i, name in enumerate(class_names)}
    assert class_name_to_id['__ignore__'] == -1
    assert class_name_to_id['_background_'] == 0
    class_names = tuple(class_names)

    # get path
    imgs_path = os.path.join(root_path, "imgs")
    anns_path = os.path.join(root_path, "anns_seg")
    png_path = os.path.join(root_path, "anns_png")
    assert os.path.isdir(anns_path), "anns_seg directory not exists."
    assert not os.path.isdir(png_path), "anns_png directory already exists"

    # start work
    color_map = get_color_map_list(256)
    with open(os.path.join(root_path, "all_list.txt"), "a") as f:
        for dir in find_dir(imgs_path):
            pre_dir = os.path.basename(dir)  # 获取并打印子文件夹名
            os.makedirs(os.path.join(png_path, pre_dir))
            imgs_list = [f for f in os.listdir(dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # 遍历图片列表
            for file in tqdm(imgs_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
                raw_name, extension = os.path.splitext(file)
                imgpath = f"{imgs_path}/{pre_dir}/{raw_name}{extension}"
                # check ann file
                annpath = f"{anns_path}/{pre_dir}/{raw_name}.json"
                assert os.path.isfile(annpath)
                # parse ann file
                with open(annpath, "r+", encoding="utf-8") as file:
                    data = json.load(file)
                    data["imageData"] = None
                    data["imagePath"] = f"../../imgs/{pre_dir}/{raw_name}{extension}"
                    # 保存修改后的文件
                    file.seek(0)
                    file.truncate()
                    file.write(json.dumps(data))
                # 获取图片长宽
                width, height = PIL.Image.open(imgpath).size
                assert width > 0 and height > 0
                # 生成 mask
                mask = shape2label([height, width], data['shapes'], class_name_to_id)
                # Assume label ranges [0, 255] for uint8,
                if mask.min() < 0 or mask.max() > 255:
                    raise Exception(
                        f'[{annpath}] Cannot save the pixel-wise class label as PNG. Please consider using the .npy format.'
                    )
                lbl_pil = PIL.Image.fromarray(mask.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(f"{png_path}/{pre_dir}/{raw_name}.png")
                f.write(f"imgs/{pre_dir}/{raw_name}{extension} anns_png/{pre_dir}/{raw_name}.png\n")
    with open(os.path.join(root_path, "all_list.txt"), "r") as f:
        list_train = f.readlines()
    list_test = list_train[::split_ratio]
    with open(os.path.join(root_path, "test.txt"), "a") as file:
        file.writelines(list_test)
    del list_train[::split_ratio]
    with open(os.path.join(root_path, "train.txt"), "a") as file:
        file.writelines(list_train)


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == "__main__":
    main(os.getcwd(), 5)
