import os


def remove_empty_folders(path):
    # 遍历指定路径下的所有文件夹
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # 递归检查文件夹是否为空
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f'Deleted empty folder: {dir_path}')

    # 再次检查顶层文件夹是否为空
    if not os.listdir(path):
        os.rmdir(path)
        print(f'Deleted empty folder: {path}')


# 示例用法
remove_empty_folders(os.getcwd())
