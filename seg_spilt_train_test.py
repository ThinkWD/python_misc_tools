import os
import shutil


# 遍历目录得到目录下的子文件夹
def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


# 遍历目录得到所有文件
def find_files(path):
    return [item.path for item in os.scandir(path) if item.is_file()]


def process(root_path, split_ratio, task):
    if task == "seg":
        # 遍历 root_path 下的子文件夹
        dirs = find_dir(root_path)
        os.mkdir(f"{root_path}/anns")
        os.mkdir(f"{root_path}/imgs")
        for dir in dirs:
            # 准备路径
            pre_dir = os.path.basename(dir)
            src_anns_dir = f"{dir}/train/anns_png"
            src_imgs_dir = f"{dir}/train/imgs"
            dst_anns_dir = f"{root_path}/anns/{pre_dir}"
            dst_imgs_dir = f"{root_path}/imgs/{pre_dir}"
            os.mkdir(dst_anns_dir)
            os.mkdir(dst_imgs_dir)
            # 获取目录下所有文件列表
            list_train = os.listdir(src_anns_dir)
            # 测试集
            list_test = list_train[::split_ratio]
            with open(os.path.join(root_path, "test.txt"), "a") as file:
                for filename in list_test:
                    filename = filename.rstrip(".png")
                    shutil.copy(
                        f"{src_anns_dir}/{filename}.png",
                        f"{dst_anns_dir}/{filename}.png",
                    )
                    shutil.copy(
                        f"{src_imgs_dir}/{filename}.jpg",
                        f"{dst_imgs_dir}/{filename}.jpg",
                    )
                    file.write(f"{pre_dir}/{filename}\n")
            # 训练集
            del list_train[::split_ratio]
            with open(os.path.join(root_path, "train.txt"), "a") as file:
                for filename in list_train:
                    filename = filename.rstrip(".png")
                    shutil.copy(
                        f"{src_anns_dir}/{filename}.png",
                        f"{dst_anns_dir}/{filename}.png",
                    )
                    shutil.copy(
                        f"{src_imgs_dir}/{filename}.jpg",
                        f"{dst_imgs_dir}/{filename}.jpg",
                    )
                    file.write(f"{pre_dir}/{filename}\n")

    elif task == "xml":
        # 检测目录是否存在
        anns_dir = os.path.join(root_path, "anns")
        imgs_dir = os.path.join(root_path, "imgs")
        assert os.path.exists(anns_dir) and os.path.isdir(anns_dir)
        assert os.path.exists(imgs_dir) and os.path.isdir(imgs_dir)
        # 创建文件夹
        anns_obj_dir = f"{root_path}/test/anns/"
        imgs_obj_dir = f"{root_path}/test/imgs/"
        assert not os.path.exists(anns_obj_dir)
        assert not os.path.exists(imgs_obj_dir)
        os.makedirs(anns_obj_dir)
        os.makedirs(imgs_obj_dir)
        # 获取xml文件列表
        list_train = os.listdir(anns_dir)
        # 训练集
        list_test = list_train[::split_ratio]
        for file in list_test:
            # 移动文件
            shutil.move(f"{anns_dir}/{file}", f"{anns_obj_dir}/{file}")
            shutil.move(
                f"{imgs_dir}/{file[:-4]}.png", f"{imgs_obj_dir}/{file[:-4]}.png"
            )


# python labelme_converter.py ./anns ./imgs ./ --tasks det
def main():
    # root path
    root_path = os.getcwd()
    split_ratio = 10
    process(root_path, split_ratio, "seg")

    print("\nAll process success\n")


if __name__ == "__main__":
    main()
