#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=4,5,6,7
export nproc_per_node=4
export PYTHONPATH=/workspace:$PYTHONPATH

mkdir -p logs/dpo/
setting_num=1

torchrun --nproc_per_node=$nproc_per_node \
src/dpo/train.py \
    --config_file=train_configs/dpo/settings_${setting_num}.yaml \
2>&1 | tee logs/dpo/settings_${setting_num}.log
        