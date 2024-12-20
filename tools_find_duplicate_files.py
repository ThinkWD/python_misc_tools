import os

from tqdm import tqdm


def find_duplicate_files(folder_path):
    file_dict = {}
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files, desc=f'{root}\t', leave=True, ncols=100, colour='CYAN'):
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file)
            if file_name in file_dict:
                file_dict[file_name].append(file_path)
            else:
                file_dict[file_name] = [file_path]

    with open('./duplicate.txt', 'w', encoding='utf-8') as f:
        for file_name, file_paths in file_dict.items():
            if len(file_paths) > 1:
                f.write(f"Duplicate file '{file_name}':\n")
                for file_path in file_paths:
                    f.write(file_path + '\n')


find_duplicate_files(os.getcwd())
