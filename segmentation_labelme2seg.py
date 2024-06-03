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


def data_shapes_rotate(shapes, width, height):
    x_scale = height / width
    y_scale = width / height
    for shape in shapes:
        for pt in shape['points']:
            x_tmp = width - pt[0]
            pt[0] = pt[1] * y_scale
            pt[1] = x_tmp * x_scale


def main(root_path, split_ratio, format="paddle"):
    # init class_names
    class_names = ['__ignore__', '_background_', 'belt_L', 'belt_R', 'roller_L', 'roller_R']  # 0: 刻度, 1: 指针
    class_name_to_id = {name: i - 1 for i, name in enumerate(class_names)}
    assert class_name_to_id['__ignore__'] == -1
    assert class_name_to_id['_background_'] == 0
    class_names = tuple(class_names)
    color_map = get_color_map_list(256)

    # get path
    imgs_path = os.path.join(root_path, "imgs")
    anns_path = os.path.join(root_path, "anns_seg")
    png_path = os.path.join(root_path, "anns_png")
    assert os.path.isdir(anns_path), "anns_seg directory not exists."
    assert not os.path.isdir(png_path), "anns_png directory already exists"

    # start work
    dataset = []
    for dir in find_dir(imgs_path):
        prefix = os.path.basename(dir)
        if prefix.endswith('_rotated'):
            print(f"Easy to confuse directories: {prefix}, continue.")
            continue
        rotate_mode = True if prefix.endswith('_R') else False
        if not rotate_mode:
            os.makedirs(os.path.join(png_path, prefix))
        else:
            new_prefix = f"{prefix[:-1]}rotated"
            os.makedirs(os.path.join(imgs_path, new_prefix), exist_ok=True)
            os.makedirs(os.path.join(anns_path, new_prefix), exist_ok=True)
            os.makedirs(os.path.join(png_path, new_prefix))
        used_prefix = new_prefix if rotate_mode else prefix
        imgs_list = [f for f in os.listdir(dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 遍历图片列表
        for file in tqdm(imgs_list, desc=f"{prefix}\t", leave=True, ncols=100, colour="CYAN"):
            raw_name, extension = os.path.splitext(file)
            imgpath = f"{imgs_path}/{prefix}/{raw_name}{extension}"
            # check image size
            cur_image = PIL.Image.open(imgpath)
            width, height = cur_image.size
            assert width > 0 and height > 0
            if rotate_mode:
                cur_image = cur_image.rotate(90, expand=True)
                cur_image = cur_image.resize((width, height))
                cur_image.save(f'{imgs_path}/{new_prefix}/{raw_name}{extension}')
            # check ann file
            annpath = f"{anns_path}/{prefix}/{raw_name}.json"
            assert os.path.isfile(annpath)
            # parse ann file
            with open(annpath, "r+", encoding="utf-8") as file:
                data = json.load(file)
                assert width == data["imageWidth"] and height == data["imageHeight"]
                data["imageData"] = None
                data["imagePath"] = f"../../imgs/{prefix}/{raw_name}{extension}"
                file.seek(0)
                file.truncate()
                file.write(json.dumps(data, indent=4))
            if rotate_mode:
                data_shapes_rotate(data['shapes'], width, height)
                data["imagePath"] = f"../../imgs/{new_prefix}/{raw_name}{extension}"
                with open(f'{anns_path}/{new_prefix}/{raw_name}.json', "w") as new_file:
                    new_file.write(json.dumps(data, indent=4))
            # generate mask
            mask = shape2label([height, width], data['shapes'], class_name_to_id)
            # Assume label ranges [0, 255] for uint8,
            if mask.min() < 0 or mask.max() > 255:
                raise Exception(f'[{annpath}] Cannot save the pixel-wise class label as PNG.')
            lbl_pil = PIL.Image.fromarray(mask.astype(np.uint8), mode='P')
            lbl_pil.putpalette(color_map)
            lbl_pil.save(f"{png_path}/{used_prefix}/{raw_name}.png")
            if format == "paddle":
                dataset.append(f"imgs/{used_prefix}/{raw_name}{extension} anns_png/{used_prefix}/{raw_name}.png\n")
            elif format == "mmlab":
                dataset.append(f"{used_prefix}/{raw_name}\n")
            else:
                raise Exception("Only support Paddle OCR format and mmlab OCR format")

    with open(os.path.join(root_path, "all_list.txt"), "w", encoding='utf-8') as file:
        file.writelines(dataset)
    test_data = dataset[::split_ratio]
    with open(os.path.join(root_path, "test.txt"), "w", encoding='utf-8') as file:
        file.writelines(test_data)
    del dataset[::split_ratio]
    with open(os.path.join(root_path, "train.txt"), "w", encoding='utf-8') as file:
        file.writelines(dataset)


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == "__main__":
    main(os.getcwd(), 5)
