import os
import json
from tqdm import tqdm

# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 遍历目录得到所有文件
def find_files(path):
    return [item.path for item in os.scandir(path) if item.is_file()]


# 修改 json 内容
def change_json(jsonfile):
    with open(jsonfile, "r") as file_in:
        json_data = json.load(file_in)
    json_data["imageData"] = None
    imagePath = json_data["imagePath"]
    pos = imagePath.rfind("\\")
    if pos > 0:
        imageName = imagePath[pos + 1 :]
    else:
        pos = imagePath.rfind("/")
        imageName = imagePath[pos + 1 :] if pos > 0 else imagePath
    json_data["imagePath"] = f"../imgs/{imageName}"
    with open(jsonfile, "w") as file_out:
        file_out.write(json.dumps(json_data))


def process(root_path):
    # 获取 ann 文件列表
    ann_list = os.listdir(root_path)
    # 设置 tqdm 进度条(BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)
    with tqdm(
        total=len(ann_list), desc="", leave=True, ncols=100, colour="CYAN"
    ) as pbar:
        # 遍历 ann 文件列表
        for file in ann_list:
            file = file.strip()
            houzhui = file[file.rindex(".") + 1 :]
            if houzhui == "json":
                change_json(os.path.join(root_path, file))
            pbar.update(1)


# python labelme_converter.py ./anns ./imgs ./ --tasks det
def main():
    # root path
    root_path = os.path.join(os.getcwd(), "anns")

    process(root_path)

    print("\nAll process success\n")


if __name__ == "__main__":
    main()
