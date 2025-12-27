#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
export nproc_per_node=4
export PYTHONPATH=/workspace:$PYTHONPATH

mkdir -p logs/

for setting_num in $(seq 1 1); do

    torchrun --nproc_per_node=$nproc_per_node \
    src/train.py \
        --config_file=train_configs/settings_${setting_num}.yaml \
    2>&1 | tee logs/settings_${setting_num}.log
        
done