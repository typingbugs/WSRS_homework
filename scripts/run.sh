#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
export nproc_per_node=4
export PYTHONPATH=/workspace:$PYTHONPATH

mkdir -p logs/

for setting_num in $(seq 2 2); do

    mkdir -p logs/settings_${setting_num}

    torchrun --nproc_per_node=$nproc_per_node \
    src/train.py \
        --config_file=train_configs/settings_${setting_num}/stage_1.yaml \
    2>&1 | tee logs/settings_${setting_num}/stage_1.log

    torchrun --nproc_per_node=$nproc_per_node \
    src/train.py \
        --config_file=train_configs/settings_${setting_num}/stage_2.yaml \
    2>&1 | tee logs/settings_${setting_num}/stage_2.log
        
done