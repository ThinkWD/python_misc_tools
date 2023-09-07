import os
from tqdm import tqdm

root_path = os.getcwd()


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 创建 coco
def voc_convert(workpath, split_ratio):
    print(f"\n[info] start task...")
    path = os.path.join(workpath, "src/imgs")
    dirs = find_dir(path)
    with open("all_list.txt", "a") as f:
        for dir in dirs:
            pre_dir = os.path.basename(dir)  # 获取并打印子文件夹名
            img_list = [f for f in os.listdir(dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for file in tqdm(img_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
                name, _ = os.path.splitext(os.path.basename(file))
                f.write(f"{pre_dir}/{name}\n")
    with open("all_list.txt", "r") as f:
        list_train = f.readlines()
    list_test = list_train[::split_ratio]
    with open("test.txt", "a") as file:
        file.writelines(list_test)
    del list_train[::split_ratio]
    with open("train.txt", "a") as file:
        file.writelines(list_train)


if __name__ == "__main__":
    assert os.path.isdir(os.path.join(root_path, "src"))
    voc_convert("src", 20)
    print("\nAll process success\n")
