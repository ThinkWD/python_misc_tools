#!/bin/bash

task=$1
config_file=$2
model_file=$3
output_file=$4
output_name=$5

# check file name
filename=$(basename -- "$config_file")
extension="${filename##*.}"
filename="${filename%.*}"
tempdir=./.tmp_conver2onnx

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 task config_file model_file save_path output_name"
    echo "task: VStreamer or MDetector"
    echo "output_name: ppyoloe-s=p2o.Mul.157; ppyoloe-osd=p2o.Mul.281."
    exit 1
fi

if [ ! -f "$config_file" ] || [[ ! "$config_file" =~ \.yml$ ]]; then
    echo "config_file: 文件不存在或不是 yml 文件"
    exit 1
fi

if [ ! -f "$model_file" ] || [[ ! "$model_file" =~ \.pdparams$ ]]; then
    echo "model_file: 文件不存在或不是 pdparams 文件"
    exit 1
fi

function VStreamer()
{
    # 临时目录用于放中间文件
    mkdir $tempdir

    # 导出推理模型
    python3 /home/lx_dir/paddle/paddledet/tools/export_model.py \
            -c $config_file \
            -o weights=$model_file exclude_nms=True trt=True \
            --output_dir $tempdir

    # 将推理模型转为 onnx
    paddle2onnx --model_dir $tempdir/$filename \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --opset_version 11 \
                --save_file $tempdir/$filename/$filename.onnx

    # 裁剪 onnx, 删除 scale factor
    python3 /home/lx_dir/paddle/Paddle2ONNX/tools/onnx/prune_onnx_model.py \
            --model $tempdir/$filename/$filename.onnx \
            --output_names concat_14.tmp_0 $output_name \
            --save_file $output_file

    rm -rf $tempdir
}

function MDetector()
{
    # 临时目录用于放中间文件
    mkdir $tempdir

    # 导出推理模型
    python3 /home/lx_dir/paddle/paddledet/tools/export_model.py \
            -c $config_file \
            -o weights=$model_file \
            --output_dir $tempdir

    # 将推理模型转为 onnx
    paddle2onnx --model_dir $tempdir/$filename \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --opset_version 11 \
                --save_file $output_file

    rm -rf $tempdir
}

# check task type
case "$task" in
    "VStreamer")
        VStreamer
        ;;
    "MDetector")
        MDetector
        ;;
    *)
        echo "Unknow task: $task"
        exit 1
        ;;
esac