from datasets import Dataset
import json
from dataclasses import dataclass
import torch
from typing import List, Dict
from pathlib import Path


def init_dataset(data_args, tokenizer):
    data_dir = Path(data_args.data_dir)
    train_dataset_path = data_dir / "dpo_train_dataset.jsonl"
    validation_dataset_path = data_dir / "dpo_valid_dataset.jsonl"

    train_raw_data = {"prompt": [], "chosen": [], "rejected": []}
    with open(train_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            datum = json.loads(line)
            train_raw_data['prompt'].append(datum['prompt'])
            train_raw_data['chosen'].append(datum['chosen'])
            train_raw_data['rejected'].append(datum['rejected'])

    validation_raw_data = {"prompt": [], "chosen": [], "rejected": []}
    with open(validation_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            datum = json.loads(line)
            validation_raw_data['prompt'].append(datum['prompt'])
            validation_raw_data['chosen'].append(datum['chosen'])
            validation_raw_data['rejected'].append(datum['rejected'])
    
    train_dataset = Dataset.from_dict(train_raw_data)
    validation_dataset = Dataset.from_dict(validation_raw_data)

    # def preprocess(example):
    #     model_inputs = {
    #         "chosen": example['prompt'] + example['chosen'],
    #         "rejected": example['prompt'] + example['rejected'],
    #     }
    #     return model_inputs
    
    # train_dataset = train_dataset.map(
    #     function=preprocess,
    #     remove_columns=["prompt"],
    #     desc="Running tokenizer on train dataset",
    # )

    # validation_dataset = validation_dataset.map(
    #     function=preprocess,
    #     remove_columns=["prompt"],
    #     desc="Running tokenizer on validation dataset",
    # )

    print(train_dataset[0])
    return train_dataset, validation_dataset
