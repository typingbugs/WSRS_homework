import pandas as pd
import ast
from typing import List, Dict, Any
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt


def load_csv(csv_path: str) -> List[Dict[str, Any]]:
    raw_data = pd.read_csv(
        csv_path,
        converters={
            "history_item_id": ast.literal_eval,
            "history_item_title": ast.literal_eval
        }
    )
    dataset = raw_data.to_dict(orient="records")
    return dataset

def save_json(data, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_jsonl(data: List[Dict[str, Any]], save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(json_path: str) -> Dict[str, int] | Dict[str, str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_item_mappings(test_dataset_raw: List[Dict[str, Any]]) -> None:
    id2item: Dict[int, str] = {}

    for example in tqdm(test_dataset_raw):
        item_ids = example['history_item_id']
        item_names = example['history_item_title']
        for item_id, item_name in zip(item_ids, item_names):
            if id2item.get(item_id, '') == '':
                id2item[item_id] = item_name

    item2id = {v: k for k, v in id2item.items()}
    id2item = {v: k for k, v in item2id.items()}
    
    print(f"Found unique items: {len(id2item)}")

    for item_id in range(1, 1 + 17408):
        if id2item.get(item_id, '') == '':
            id2item[item_id] = f'unknown_item_{item_id}'

    id2item = dict(sorted(id2item.items(), key=lambda x: x[0]))

    save_json(id2item, 'data_preprocess/outputs/id2item.json')
    item2id = {v: k for k, v in id2item.items()}
    print(len(item2id))
    save_json(item2id, 'data_preprocess/outputs/item2id.json')


def plot_length_freq(lengths: List[int], save_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Length of "history_item_id"')
    plt.ylabel('Frequency (log10 scale)')
    plt.title('Distribution of "history_item_id" Lengths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {save_path}")


def split_dataset(test_dataset_raw: List[Dict[str, Any]], split_ratio: float = 0.2):
    all_data = []

    for example in tqdm(test_dataset_raw):
        all_data.append({
            'user_id': example['user_id'],
            'history_item_id': example['history_item_id']
        })

    # Calculate lengths of history_item_id
    lengths = [len(example['history_item_id']) for example in all_data]
    avg_length = np.mean(lengths)
    print(f"Average title sequence length in test data: {avg_length}")

    plot_length_freq(
        lengths, 
        save_path='data_preprocess/outputs/len_freq.png'
    )

    valid_indices = random.sample(range(len(all_data)), k=int(split_ratio * len(all_data)))
    valid_set = [all_data[i] for i in valid_indices]
    train_set = [all_data[i] for i in range(len(all_data)) if i not in valid_indices]
    inference_set = all_data

    print(f"Train set size: {len(train_set)}, Validation set size: {len(valid_set)}")

    save_jsonl(data=train_set, save_path="data_preprocess/outputs/train_dataset.jsonl")
    save_jsonl(data=valid_set, save_path="data_preprocess/outputs/valid_dataset.jsonl")
    save_jsonl(data=inference_set, save_path="data_preprocess/outputs/inference_dataset.jsonl")



def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    test_dataset_raw: List[Dict[str, Any]] = load_csv(csv_path=dataset_path)
    build_item_mappings(test_dataset_raw)
    split_dataset(test_dataset_raw, split_ratio=0.1)


if __name__ == "__main__":
    random.seed(42)
    load_dataset(dataset_path="data/raw_data/test2.csv")