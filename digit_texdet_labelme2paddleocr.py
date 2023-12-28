import os
import json
from tqdm import tqdm


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 解析单个 labelme 标注文件(json)
def parse_labelme(json_data):
    # 遍历 shapes
    anns = []
    shapes = json_data["shapes"]
    for shape in shapes:
        # 检查 shape_type
        shape_type = shape["shape_type"]
        if shape_type != "polygon":
            print("Only 'polygon' boxes are supported.")
            return {}
        # 读取 points
        points = []
        points_xy = shape["points"]
        for point in points_xy:
            points.extend([int(x) for x in point])
        if len(points) < 8 or len(points) % 2 != 0:
            print(f"Invalid polygon: {json_data}.")
            return {}
        s = []
        for i in range(0, len(points), 2):
            b = points[i:i + 2]
            b = [int(t) for t in b]
            s.append(b)
        ann = {"transcription": shape["label"], "points": s}
        anns.append(ann)
    return anns


def check_ann_file(annpath):
    try:
        if not os.path.isfile(annpath):
            raise FileNotFoundError("file not exists!")
        with open(annpath, "r", encoding="utf-8") as file:
            data = json.load(file)
        if data["imageData"] != None:
            data["imageData"] = None
            image_name = os.path.basename(data["imagePath"])
            data["imagePath"] = f"../imgs/{image_name}"
            with open(annpath, "w") as file_out:
                file_out.write(json.dumps(data))
        return parse_labelme(data)
    except Exception as e:
        print(f"\n\n{e}\n\n")
        exit(0)


def process(root_path, task):
    print(f"\n[info] start task {task}...")
    with open(f"./{task}.txt", "w") as ann_file:
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
                imgpath = f"{task}/{pre_dir}/imgs/{raw_name}{extension}"
                annpath = f"./{task}/{pre_dir}/anns_seg/{raw_name}.json"
                # 解析单个 ann 文件
                ann = check_ann_file(annpath)
                ann_file.write(imgpath + '\t' + json.dumps(ann, ensure_ascii=False) + '\n')
        


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
