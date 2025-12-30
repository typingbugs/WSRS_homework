#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=4,5,6,7
export nproc_per_node=4
export PYTHONPATH=/workspace:$PYTHONPATH

mkdir -p logs/sft/

setting_num=2

mkdir -p logs/sft/settings_${setting_num}

torchrun --nproc_per_node=$nproc_per_node \
src/sft/train.py \
    --config_file=train_configs/sft/settings_${setting_num}/stage_1.yaml \
2>&1 | tee logs/sft/settings_${setting_num}/stage_1.log

torchrun --nproc_per_node=$nproc_per_node \
src/sft/train.py \
    --config_file=train_configs/sft/settings_${setting_num}/stage_2.yaml \
2>&1 | tee logs/sft/settings_${setting_num}/stage_2.log
        