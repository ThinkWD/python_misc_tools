import os
import json
import shutil
from tqdm import tqdm

##################################################################
#
#   此文件用于批量重命名数据集, 命名由 offset 开始递增.
#
#   ./imgs      --> ./rename/imgs
#   ./anns      --> ./rename/anns
#   ./anns_seg  --> ./rename/anns_seg
#
##################################################################


offset = 0


def rename(root_path):
    assert os.path.isdir(f"{root_path}/imgs"), "图片文件夹不存在!"
    assert not os.path.isdir(f"{root_path}/rename"), "目标文件夹已存在!"
    os.makedirs(f"{root_path}/rename/imgs")
    if os.path.isdir(f"{root_path}/anns"):
        os.makedirs(f"{root_path}/rename/anns")
    if os.path.isdir(f"{root_path}/anns_seg"):
        os.makedirs(f"{root_path}/rename/anns_seg")
    # get images list
    imgs_list = [f for f in os.listdir(f"{root_path}/imgs") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    namebit = 6 if len(imgs_list) > 9999 else 4
    # prosess
    for idx, file in enumerate(tqdm(imgs_list, leave=True, ncols=100, colour="CYAN")):
        raw_name, extension = os.path.splitext(file)
        out_name = str(idx + offset).zfill(namebit)
        # img
        shutil.copy(f"{root_path}/imgs/{file}", f"{root_path}/rename/imgs/{out_name}{extension}")
        # ann det
        det_file = f"{root_path}/anns/{raw_name}.xml"
        if os.path.isfile(det_file):
            shutil.copy(det_file, f"{root_path}/rename/anns/{out_name}.xml")
        # ann seg
        seg_file = f"{root_path}/anns_seg/{raw_name}.json"
        if os.path.isfile(seg_file):
            with open(seg_file, "r", encoding='utf-8') as file_in:
                json_data = json.load(file_in)
            json_data["imageData"] = None
            json_data["imagePath"] = f"../imgs/{out_name}{extension}"
            with open(f"{root_path}/rename/anns_seg/{out_name}.json", "w", encoding='utf-8') as file_out:
                json.dump(json_data, file_out, indent=4)


def main():
    rename(os.getcwd())
    print("\nAll process success\n")


if __name__ == "__main__":
    main()
