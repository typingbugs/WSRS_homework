from datasets import Dataset, concatenate_datasets
import json
from dataclasses import dataclass
import torch
from typing import List, Dict
from pathlib import Path


def init_dataset(data_args, tokenizer):
    data_dir = Path(data_args.data_dir)
    train_dataset_path = data_dir / "train_dataset.jsonl"
    validation_dataset_path = data_dir / "valid_dataset.jsonl"

    train_raw_data = {"history_item_id": []}
    with open(train_dataset_path, 'r', encoding='utf-8') as f:
        train_raw_data['history_item_id'] = [json.loads(line)['history_item_id'] for line in f]
    validation_raw_data = {"history_item_id": []}
    with open(validation_dataset_path, 'r', encoding='utf-8') as f:
        validation_raw_data['history_item_id'] = [json.loads(line)['history_item_id'] for line in f]
    
    train_dataset = Dataset.from_dict(train_raw_data)
    validation_dataset = Dataset.from_dict(validation_raw_data)

    def preprocess_remove_last(example):
        input_ids = [tokenizer.bos_token_id] + example['history_item_id']
        input_ids = input_ids[:-1]
        model_inputs = {
            'input_ids': input_ids[:-1],
            'labels': input_ids[1:]
        }
        return model_inputs
    
    def preprocess(example):
        input_ids = [tokenizer.bos_token_id] + example['history_item_id']
        model_inputs = {
            'input_ids': input_ids[:-1],
            'labels': input_ids[1:]
        }
        return model_inputs
    
    train_dataset = train_dataset.map(
        function=preprocess,
        remove_columns=['history_item_id'],
        desc="Running tokenizer on train dataset",
    )
    validation_dataset_for_train = validation_dataset.map(
        function=preprocess_remove_last,
        remove_columns=['history_item_id'],
        desc="Running tokenizer on validation dataset for training",
    )
    train_dataset = concatenate_datasets([train_dataset, validation_dataset_for_train])

    validation_dataset = validation_dataset.map(
        function=preprocess,
        remove_columns=['history_item_id'],
        desc="Running tokenizer on validation dataset",
    )

    # Filter out empty sequences
    train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) > 0)
    validation_dataset = validation_dataset.filter(lambda x: len(x['input_ids']) > 0)

    print(train_dataset[0])
    return train_dataset, validation_dataset


@dataclass
class RecommendDataCollator:
    def __init__(self, pad_token_id: int = 0, model_max_length: int = 768):
        self.pad_token_id = pad_token_id
        self.model_max_length = model_max_length

    def __call__(self, features: List[Dict[str, List[int]]]):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # batch max length
        max_len = min(self.model_max_length, max(len(x) for x in input_ids))

        batch_input = []
        batch_labels = []
        batch_masks = []

        for input_id, label in zip(input_ids, labels):
            if len(input_id) > max_len:
                input_id = input_id[-max_len:]
                label = label[-max_len:]
            padding_length = max_len - len(input_id)
            batch_input.append(input_id + [self.pad_token_id] * padding_length)
            batch_masks.append([1] * len(input_id) + [0] * padding_length)
            batch_labels.append(label + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }