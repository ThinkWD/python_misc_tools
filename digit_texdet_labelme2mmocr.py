import os
import json
import warnings
import cv2
from tqdm import tqdm


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 遍历目录得到所有文件
def find_files(path):
    return [item.path for item in os.scandir(path) if item.is_file()]


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
            msg = f"Only 'rectangle' and 'polygon' boxes are supported. Boxes with {shape} will be discarded."
            warnings.warn(msg)
            return {}
        # 读取 points
        points_x1y1 = []
        points_xy = shape["points"]
        for point in points_xy:
            points_x1y1.extend([float(x) for x in point])
        x_list = points_x1y1[::2]
        y_list = points_x1y1[1::2]
        # 根据 shape_type 做不同处理
        quad = []
        if shape_type == "rectangle":
            quad = [
                points_x1y1[0],
                points_x1y1[1],
                points_x1y1[2],
                points_x1y1[1],
                points_x1y1[2],
                points_x1y1[3],
                points_x1y1[0],
                points_x1y1[3],
            ]
        else:
            if len(points_x1y1) < 8 or len(points_x1y1) % 2 != 0:
                msg = f"Invalid polygon {points_x1y1}. The polygon is expected to have 8 or more than 8 even number of coordinates in MMOCR."
                warnings.warn(msg)
                return {}
            if len(points_x1y1) == 8:
                quad = points_x1y1
            else:
                x_min, x_max, y_min, y_max = (
                    min(x_list),
                    max(x_list),
                    min(y_list),
                    max(y_list),
                )
                quad = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

        text_label = shape["label"]
        w = max(x_list) - min(x_list)
        h = max(y_list) - min(y_list)
        ann = {
            "iscrowd": 0 if text_label != "###" else 1,
            "category_id": 1,
            "bbox": [min(x_list), min(y_list), w, h],
            "segmentation": [quad] if shape_type == "rectangle" else [points_x1y1],
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
    try:
        if not os.path.exists(jsonfile):
            raise Exception("")
        data = json.load(open(jsonfile, "r", encoding="utf-8"))
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


def process(root_path, task_type):
    # 创建 mmocr-dbnet 格式的基本结构
    anns = []
    # 遍历 root_path 下的子文件夹
    dirs = find_dir(root_path)
    for dir in dirs:
        # 获取并打印子文件夹名
        pre_dir = os.path.basename(dir)
        print("\n  >>  " + pre_dir)
        # 获取 img 文件列表
        img_list = os.listdir(os.path.join(dir, "imgs"))
        # 设置 tqdm 进度条(BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)
        with tqdm(
            total=len(img_list), desc="", leave=True, ncols=100, colour="CYAN"
        ) as pbar:
            # 遍历 ann 文件列表
            for file in img_list:
                # 获取文件名(带多文件夹的相对路径)
                file = file.strip()
                imgname = f"{task_type}/{pre_dir}/imgs/{file}"
                annpath = os.path.join(
                    root_path, f"{pre_dir}/anns_seg/{file[:-4]}.json"
                )
                # 解析单个 ann 文件
                ann = check_ann_file(annpath, imgname)
                # 更新 anns
                anns.append(ann)
                # 更新进度条
                pbar.update(1)

    # 导出并保存到 txt 文件
    with open(f"{task_type}.txt", "w") as file:
        file.write("\n".join(anns))


# 图片文件夹：imgs
# 标签文件夹：anns_seg
def main():
    # root path
    root_path = os.getcwd()

    # 根据建立的文件夹判断要进行哪些任务
    train_dir = f"{root_path}/train"
    if os.path.exists(train_dir) and os.path.isdir(train_dir):
        print("\n[info] task : train...")
        process(train_dir, "train")
    else:
        print("\n[error] 检查路径是否正确")
        exit(0)

    test_dir = f"{root_path}/test"
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        print("\n[info] task : test...")
        process(test_dir, "test")

    val_dir = f"{root_path}/val"
    if os.path.exists(val_dir) and os.path.isdir(val_dir):
        print("\n[info] task : val...")
        process(val_dir, "val")

    print("\nAll process success\n")


if __name__ == "__main__":
    main()
