from typing import Dict
import numpy as np
import json
import os
from collections import defaultdict
from data_preprocess.build_pretrained_embeddings.embedding_models.encode import encode


def load_jsonl(jsonl_path: str):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def load_json(json_path: str) -> Dict[str, int] | Dict[str, str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


max_item_id = 17408


def main(setting: str, model_name: str, backend: str, num_workers: int = 4, batch_size: int = 64, truncate_dim: int = None):
    item_infos: Dict[str, str] = load_jsonl(jsonl_path="data_preprocess/build_pretrained_embeddings/outputs/merged_information.jsonl")
    item2id: Dict[str, str] = load_json(json_path="data_preprocess/outputs/item2id.json")

    known_item_info = []
    item_to_embed_indice = []
    for item in item_infos:
        item_title = item['item_title']
        item_info = item["info"] if setting != "naive" else item_title
        item_id = item["item_id"]
        if item_title != f"unknown_item_{item_id}":
            item_to_embed_indice.append(len(known_item_info))
            known_item_info.append(item_info)
        else:
            item_to_embed_indice.append(-1)

    embeddings = encode(
        model_path=model_name,
        texts=known_item_info,
        batch_size=batch_size,
        truncate_dim=truncate_dim,
        num_workers=num_workers,
        backend=backend,
    )

    embedding_dim = embeddings.shape[1]
    print(f"Embedding dim = {embedding_dim}")

    full_embedding = np.zeros((max_item_id, embedding_dim), dtype=np.float32)

    train_seqs = []
    with open("data/cleaned_data/train_dataset.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_seq = [item2id.get(title, 0) for title in data['title_sequence']]
            train_seqs.append(data_seq)
    # 构建倒排索引
    item_neighbors = defaultdict(list)
    for seq in train_seqs:
        seq = [x for x in seq if x != 0]
        for i, item in enumerate(seq):
            for nb in seq:
                if nb != item:
                    item_neighbors[item].append(nb)

    for idx, item_to_embed_index in enumerate(item_to_embed_indice):
        item_id = item_infos[idx]['item_id']
        array_idx = item_id - 1  # item_id 从 1 开始，数组索引从 0 开始
        
        if item_to_embed_index != -1:
            full_embedding[array_idx] = embeddings[item_to_embed_index]

    unknown_count = 0
    for idx, item_to_embed_index in enumerate(item_to_embed_indice):
        item_id = item_infos[idx]['item_id']
        array_idx = item_id - 1  # item_id 从 1 开始，数组索引从 0 开始   

        if item_to_embed_index == -1:
            unknown_count += 1
            neighbors = item_neighbors.get(item_id, [])
            if len(neighbors) == 0:
                full_embedding[array_idx] = np.random.normal(0, 1, embedding_dim).astype(np.float32)
                continue
            neighbor_embs = full_embedding[np.array(neighbors) - 1]
            mask = np.any(neighbor_embs != 0, axis=1)
            neighbor_embs = neighbor_embs[mask]
            if neighbor_embs.shape[0] == 0:
                full_embedding[array_idx] = np.random.normal(0, 1, embedding_dim).astype(np.float32)
                continue
            mean_emb = neighbor_embs.mean(axis=0)
            full_embedding[array_idx] = mean_emb.astype(np.float32)
    print(f"Total unknown items: {unknown_count}")

    # 归一化
    norms = np.linalg.norm(full_embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    full_embedding = full_embedding / norms

    output_file = os.path.join("data/item_embeddings", f"{setting}_{model_name.split('/')[-1]}.npy")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, full_embedding)
    print(f"Saved embeddings to {output_file}, shape={full_embedding.shape}")


if __name__ == "__main__":
    num_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_workers = len(num_gpus.split(",")) if num_gpus else 1
    batch_size = 64
    truncate_dim = 2048

    settings = [
        "naive", 
        # "add-info"
    ]
    model_names = [
        {"model_name": "data/embedding_models/sentence-transformers/all-mpnet-base-v2", "backend": "mpnet"},
        {"model_name": "data/embedding_models/Qwen/Qwen3-Embedding-0.6B", "backend": "qwen"},
        {"model_name": "data/embedding_models/BAAI/bge-large-en-v1.5", "backend": "bge"},
        {"model_name": "data/embedding_models/Qwen/Qwen3-Embedding-8B", "backend": "qwen"}
    ]
    for setting in settings:
        for model in model_names:
            print(f"Processing setting: {setting}, model: {model['model_name']}")
            main(setting, **model, num_workers=num_workers, batch_size=batch_size, truncate_dim=truncate_dim)
