import os
import json
import labelme
from tqdm import tqdm


def find_dir(path):
    return [item.path for item in os.scandir(path) if item.is_dir()]


def main(root_path, split_ratio):
    # init class_names
    class_names = ['__ignore__', '_background_', '0', '1']  # 0: 刻度, 1: 指针
    class_name_to_id = {name: i - 1 for i, name in enumerate(class_names)}
    assert class_name_to_id['__ignore__'] == -1
    assert class_name_to_id['_background_'] == 0
    class_names = tuple(class_names)

    # get path
    imgs_path = os.path.join(root_path, "imgs")
    anns_path = os.path.join(root_path, "anns_seg")
    png_path = os.path.join(root_path, "anns_png")
    tmp_path = os.path.join(png_path, "tempfiletotempusepleasedontchangethisfile.json")
    assert os.path.isdir(anns_path), "anns_seg directory not exists."
    assert not os.path.isdir(png_path), "anns_png directory already exists"

    # start work
    with open(os.path.join(root_path, "all_list.txt"), "a") as f:
        for dir in find_dir(imgs_path):
            pre_dir = os.path.basename(dir)  # 获取并打印子文件夹名
            os.makedirs(os.path.join(png_path, pre_dir))
            imgs_list = [f for f in os.listdir(dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # 遍历图片列表
            for file in tqdm(imgs_list, desc=f"{pre_dir}\t", leave=True, ncols=100, colour="CYAN"):
                raw_name, extension = os.path.splitext(file)
                imgpath = f"{imgs_path}/{pre_dir}/{raw_name}{extension}"
                # check ann file
                annpath = f"{anns_path}/{pre_dir}/{raw_name}.json"
                assert os.path.isfile(annpath)
                with open(annpath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    data["imageData"] = None
                    data["imagePath"] = imgpath
                    with open(tmp_path, "w") as file_out:
                        file_out.write(json.dumps(data))
                # load and save
                label_file = labelme.LabelFile(filename=tmp_path)
                img = labelme.utils.img_data_to_arr(label_file.imageData)
                cls, _ = labelme.utils.shapes_to_label(
                    img_shape=img.shape,
                    shapes=label_file.shapes,
                    label_name_to_value=class_name_to_id,
                )
                labelme.utils.lblsave(f"{png_path}/{pre_dir}/{raw_name}.png", cls)
                f.write(f"{pre_dir}/{raw_name}\n")
        os.remove(tmp_path)
    with open(os.path.join(root_path, "all_list.txt"), "r") as f:
        list_train = f.readlines()
    list_test = list_train[::split_ratio]
    with open(os.path.join(root_path, "test.txt"), "a") as file:
        file.writelines(list_test)
    del list_train[::split_ratio]
    with open(os.path.join(root_path, "train.txt"), "a") as file:
        file.writelines(list_train)


# Reference: https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py
if __name__ == "__main__":
    main(os.getcwd(), 20)
