## python3 tools/infer_rec.py -c data/ppv3rec/en_PP-OCRv3_rec.yml

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json
from tqdm import tqdm

##################################################################
#
#   !!!!!!!!!!!!!!! 此文件可能已经过时, 需要更新以适配最新版本 !!!!!!!!!!!!!!
#
#   此文件用于测试文本识别模型
#   它会遍历数据集中的图片及其对应标签文本, 对比识别结果与标签文本, 将不匹配的条目保存下来
#
##################################################################

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


def inference():
    config['Global']['pretrained_model'] = f"{config['Global']['save_model_dir']}/best_accuracy"
    global_config = config['Global']
    # build post process
    post_process_class = build_post_process(config['PostProcess'], global_config)
    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        # distillation model
        if config['Architecture']["algorithm"] in [
            "Distillation",
        ]:
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head']['name'] == 'MultiHead':  # for multi head
                    out_channels_list = {}
                    if config['PostProcess']['name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head']['out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"]['out_channels'] = char_num
        elif config['Architecture']['Head']['name'] == 'MultiHead':  # for multi head loss
            out_channels_list = {}
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head']['out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])
    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image',
                    'encoder_word_pos',
                    'gsrm_word_pos',
                    'gsrm_slf_attn_bias1',
                    'gsrm_slf_attn_bias2',
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)
    model.eval()

    # inference images
    save_path = config['Global'].get('save_res_path', "./workspace/ppv3rec/val/res.txt")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    root_path = config['Global'].get('infer_img', "./data/ppv3rec")
    label_file_path = os.path.join(root_path, 'all_list.txt')
    error_label_num = 0
    with open(label_file_path, "r") as fread, open(save_path, "w") as fwrite:
        total_fread = len(fread.readlines())
        fread.seek(0)
        for line in tqdm(fread, total=total_fread, leave=True, ncols=120, colour="CYAN"):
            line = line.strip().split('\t')
            img_path = os.path.join(root_path, line[0])
            with open(img_path, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            if config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)
                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list),
                ]
            if config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]
            if config['Architecture']['algorithm'] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positons = np.expand_dims(batch[2], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio), paddle.to_tensor(word_positons)]
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)
            if config['Architecture']['algorithm'] == "SRN":
                preds = model(images, others)
            elif config['Architecture']['algorithm'] == "SAR":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "RobustScanner":
                preds = model(images, img_metas)
            else:
                preds = model(images)
            post_result = post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0]  # + "\t" + str(post_result[0][1])
            if info != line[1]:
                error_label_num += 1
                fwrite.write(f"{line[0]}\t\t\t{info}\n")
    print(f"\n\n\nerror_label_num: {error_label_num}\n\n\n")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    inference()
