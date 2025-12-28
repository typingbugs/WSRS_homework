#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/workspace:$PYTHONPATH

python src/inference.py \
    --model_path data/outputs/setting_1/stage_2/checkpoint-540 \
    --processor_path model_configs/model_1 \
    --test_file data/run_data/inference_dataset.jsonl \
    --top_k 10 \
    --batch_size 2048 \
    --output_file data/infer/setting_1/stage_2/checkpoint-540/output.csv