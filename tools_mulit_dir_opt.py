import os
import shutil

from tqdm import tqdm


def main(refer, target):
    class file_t:
        def __init__(self, name, path):
            self.name = name
            self.path = path

    file_list = []
    for root, _, files in os.walk(target):
        for file in files:
            name, _ = os.path.splitext(file)
            path = os.path.join(os.path.relpath(root, target), file)
            file_list.append(file_t(name, path))

    result_dir = os.path.join(refer, 'result')
    # print(f"result_dir: {result_dir}")
    for file in tqdm(os.listdir(refer), leave=True, ncols=100, colour='CYAN'):
        name, _ = os.path.splitext(file)
        matching = [file for file in file_list if file.name == name]
        for match in matching:
            src = os.path.join(target, match.path)
            dst = os.path.join(result_dir, match.path)
            dir_path = os.path.dirname(dst)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # print(f"src: {src}, dst: {dst}")
            shutil.move(src, dst)


if __name__ == '__main__':
    refer_dir = 'D:/User/Desktop/data/new/result'
    target_dir = 'D:/Work/__DataSet/数字仪表/step1_categories/src/D000'
    main(refer_dir, target_dir)
