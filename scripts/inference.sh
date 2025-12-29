#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/workspace:$PYTHONPATH

model_name="dpo/setting_1/stage_2/checkpoint-162"
output_dir="data/infer/${model_name}"
mkdir -p ${output_dir}

python src/inference.py \
    --model_path data/outputs/${model_name} \
    --test_file data/run_data/inference_dataset.jsonl \
    --top_k 10 \
    --batch_size 2048 \
    --output_file ${output_dir}/output.csv