import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import faiss
from tqdm import tqdm


def load_sequences(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            seq = obj["history_item_id"]
            if len(seq) >= 2:
                data.append({
                    "user_id": obj["user_id"],
                    "history_item_id": seq
                })
    return data


def load_freq(json_path: str) -> Dict[str, int] | Dict[str, str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_jsonl(data: List[Dict[str, Any]], save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


class NegativeSampler:
    def __init__(self, model_path, item_freq: List[float], num_items=17408, top_k=3):
        self.num_items = num_items
        self.top_k = top_k
        self.item_freq = item_freq
        
        # Load embeddings and build index
        self.item_embeddings = self._load_item_embeddings(model_path)
        self.faiss_index = self._build_faiss_index()
    
    def _load_item_embeddings(self, model_path):
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.eval()
        
        # Get embedding weights: shape [vocab_size, hidden_size]
        embedding_weights = model.get_input_embeddings().weight.detach()
        
        # Item tokens are at positions 1 to num_items (position 0 is pad_token)
        item_embeddings = embedding_weights[1:self.num_items+1].cpu().numpy().astype(np.float32)
        
        print(f"Loaded item embeddings with shape: {item_embeddings.shape}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(item_embeddings)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return item_embeddings
    
    def _build_faiss_index(self):
        print("Building FAISS index...")
        dim = self.item_embeddings.shape[1]
        
        index = faiss.IndexFlatIP(dim)
        index.add(self.item_embeddings)
        
        print(f"FAISS index built with {index.ntotal} items")
        return index
    
    def sample(self, pos_item_id, user_history_set):
        # Convert to 0-indexed for array access
        pos_idx = pos_item_id - 1
        
        # Get the embedding of the positive item
        query = self.item_embeddings[pos_idx:pos_idx+1]  # Shape: [1, dim]
        
        # Search for top_k + extra to handle filtering
        search_k = self.top_k + len(user_history_set) + 1
        _, indices = self.faiss_index.search(query, search_k)
        
        # Filter out the positive item itself and items in user history
        candidates = []
        for idx in indices[0]:
            item_id = idx + 1  # Convert back to 1-indexed
            if item_id != pos_item_id and item_id not in user_history_set:
                candidates.append(item_id)
                if len(candidates) >= self.top_k:
                    break
        
        if candidates:
            rand_num = random.random()
            if rand_num < 0.3:
                candidates = candidates[:10]
            else:
                candidates = candidates[10:]
            neg = random.choice(candidates)
        else:
            neg = -1  # Fallback in case no negative found
        
        return int(neg)


def build_dpo_dataset(
    train_path, 
    valid_path, 
    train_output_path, 
    valid_output_path,
    sampler: NegativeSampler,
    id2item: Dict[int, str]
):
    print("Loading raw sequences...")
    train_seq = load_sequences(train_path)

    train_dataset = []

    for data in tqdm(train_seq, desc="Building DPO dataset"):
        seq = data["history_item_id"]
        prompt = seq[:-1]
        pos = seq[-1]
        
        # Create set for efficient lookup
        user_history_set = set(seq)

        # Sample hard negative
        neg = sampler.sample(pos, user_history_set)
        if neg == -1:
            continue  # Skip if no valid negative found
        
        data = {
            "user_id": data["user_id"],
            "prompt": ''.join([f"<{id2item[i]}>" for i in prompt]),
            "chosen": f"<{id2item[pos]}>",
            "rejected": f"<{id2item[neg]}>"
        }
        train_dataset.append(data)

    valid_dataset = []

    valid_seq = load_sequences(valid_path)

    for data in tqdm(valid_seq, desc="Building DPO dataset"):
        seq = data["history_item_id"]
        valid_prompt = seq[:-1]
        valid_pos = seq[-1]
        train_prompt = seq[:-2]
        train_pos = seq[-2]
        
        # Create set for efficient lookup
        user_history_set = set(seq[:-1])

        # Sample hard negative
        train_neg = sampler.sample(train_pos, user_history_set)
        if train_neg == -1:
            continue  # Skip if no valid negative found
        
        train_data = {
            "user_id": data["user_id"],
            "prompt": ''.join([f"<{id2item[i]}>" for i in train_prompt]),
            "chosen": f"<{id2item[train_pos]}>",
            "rejected": f"<{id2item[train_neg]}>"
        }
        train_dataset.append(train_data)

        valid_data = {
            "user_id": data["user_id"],
            "prompt": ''.join([f"<{id2item[i]}>" for i in valid_prompt]),
            "chosen": f"<{id2item[valid_pos]}>",
            "rejected": "<EOS>"  # No negative for validation
        }
        valid_dataset.append(valid_data)
    
    print(f"Total preference pairs: {len(train_dataset)}")
    save_jsonl(train_dataset, train_output_path)
    save_jsonl(valid_dataset, valid_output_path)


def load_json(json_path: str) -> Dict[int, str]:
    with open(json_path, 'r', encoding='utf-8') as f:
       data = json.load(f)
    data = {int(k): v for k, v in data.items()}
    return data


if __name__ == "__main__":
    # Paths
    train_path = "data/run_data/train_dataset.jsonl"
    valid_path = "data/run_data/valid_dataset.jsonl"
    model_path = "data/outputs/setting_1/stage_2/checkpoint-540"
    train_output_path = "data_preprocess/outputs/dpo_train_dataset.jsonl"
    valid_output_path = "data_preprocess/outputs/dpo_valid_dataset.jsonl"
    id2item_path = "data_preprocess/outputs/id2item.json"
    freq_path = "data_preprocess/outputs/item_frequency.json"

    id2item = load_json(id2item_path)
    freq = load_freq(freq_path)
    
    # Initialize sampler
    sampler = NegativeSampler(
        model_path=model_path,
        num_items=17408,
        top_k=200,
        item_freq=freq
    )
    
    # Build dataset
    ds = build_dpo_dataset(train_path, valid_path, train_output_path, valid_output_path, sampler, id2item)
