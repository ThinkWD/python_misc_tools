import os
import json
from tqdm import tqdm


##################################################################
#
#   此文件被 text_detection_label2ocr.py 取代, 将在未来删除
#
##################################################################


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_data, filename):
    # 解析
    img_width = json_data["imageWidth"]
    img_height = json_data["imageHeight"]
    # 预定义结果
    anns = []
    # 遍历 shapes
    shapes = json_data["shapes"]
    for shape in shapes:
        # 检查 shape_type
        shape_type = shape["shape_type"]
        if shape_type not in ["rectangle", "polygon"]:
            print("Only 'rectangle' and 'polygon' boxes are supported.")
            return {}
        # 读取 points
        points = []
        points_xy = shape["points"]
        for point in points_xy:
            points.extend([float(x) for x in point])
        if len(points) < 8 or len(points) % 2 != 0:
            print(f"Invalid polygon: {json_data}.")
            return {}
        x_list = points[::2]
        y_list = points[1::2]
        # 根据 shape_type 做不同处理
        quad = []
        if shape_type == "rectangle":
            quad = [points[0], points[1], points[2], points[1], points[2], points[3], points[0], points[3]]
        elif len(points) == 8:
            quad = points
        else:
            x_min, x_max, y_min, y_max = (min(x_list), max(x_list), min(y_list), max(y_list))
            quad = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

        text_label = shape["label"]
        w = max(x_list) - min(x_list)
        h = max(y_list) - min(y_list)
        ann = {
            "iscrowd": 0 if text_label != "###" else 1,
            "category_id": 1,
            "bbox": [min(x_list), min(y_list), w, h],
            "segmentation": [quad] if shape_type == "rectangle" else [points],
            "text": text_label,
        }
        anns.append(ann)

    return {
        "file_name": filename,
        "height": img_height,
        "width": img_width,
        "annotations": anns,
    }


def check_ann_file(jsonfile, filename):
    import cv2

    try:
        if not os.path.exists(jsonfile):
            raise FileNotFoundError("file not exists!")
        with open(jsonfile, "r", encoding="utf-8") as file:
            data = json.load(file)
        if data["imageData"] != None:
            data["imageData"] = None
            image_name = os.path.basename(data["imagePath"])
            data["imagePath"] = f"../imgs/{image_name}"
            with open(jsonfile, "w") as file_out:
                file_out.write(json.dumps(data))
        result = json.dumps(parse_labelme(data, filename))
        if len(result) < 30:
            print("标签文件解析错误, 用 labelme 耐心检查是否每个标注都已经连接成封闭图形！！")
            print(f"解析错误的文件: {jsonfile}")
            print(f"该文件对应的图片: {filename}")
            exit(0)
        return result
    except Exception as e:
        anns = []
        img = cv2.imread(filename)
        result = {
            "file_name": filename,
            "height": img.shape[0],
            "width": img.shape[1],
            "annotations": anns,
        }
        return json.dumps(result)


def process(root_path, task):
    print(f"\n[info] start task {task}...")
    # 创建 mmocr-dbnet 格式的基本结构
    anns = []
    # 遍历 root_path 下的子文件夹
    dirs = find_dir(root_path)
    for dir in dirs:
        # 获取并打印子文件夹名
        pre_dir = os.path.basename(dir)
        # 获取img文件列表
        img_path = os.path.join(dir, "imgs")
        assert os.path.isdir(img_path), f"图片文件夹不存在: {img_path}"
        img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 遍历 ann 文件列表
        for file in tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
            # 获取文件名(带多文件夹的相对路径)
            raw_name, extension = os.path.splitext(file)
            imgname = f"{task}/{pre_dir}/imgs/{raw_name}{extension}"
            annpath = f"./{task}/{pre_dir}/anns_seg/{raw_name}.json"
            # 解析单个 ann 文件
            ann = check_ann_file(annpath, imgname)
            # 更新 anns
            anns.append(ann)

    # 导出并保存到 txt 文件
    with open(f"{task}.txt", "w") as file:
        file.write("\n".join(anns))


# 图片文件夹：imgs
# 标签文件夹：anns_seg
def main():
    for task in ['train', 'test', 'validation']:
        task_dir = f"./{task}"
        if os.path.isdir(task_dir):
            process(task_dir, task)

    print("\nAll process success\n")


if __name__ == "__main__":
    main()
