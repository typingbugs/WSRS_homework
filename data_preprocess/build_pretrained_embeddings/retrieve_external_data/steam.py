import faiss
from ..embedding_models.encode import encode
from typing import Dict, Any
import json
import os
from typing import List
import numpy as np


def load_json(json_path: str) -> Dict[str, int] | Dict[str, str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_jsonl(data: List[Dict[str, Any]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def delete_platform_info(title: str):
    platform_info = [
        ' - Xbox', 
        ' - PlayStation',
        ' - PS',
        ' - PC', 
        ' - Nintendo',
        ' - Sega'
    ]
    for platform in platform_info:
        index = title.find(platform)
        if index != -1:
            title = title[:index]
    return title.strip()

def load_external_dataset(json_path: str) -> Dict[int, Any]:
    data = load_json(json_path)
    game_data = {}
    for game_idx, game_info in data.items():
        new_game_info = {}
        new_game_info['id'] = int(game_idx)
        new_game_info['title'] = game_info['name']
        new_game_info['release_year'] = game_info['release_date'][-4:]
        new_game_info['price'] = game_info['price']
        new_game_info['description'] = game_info['short_description']
        new_game_info['developers'] = game_info['developers']
        new_game_info['genres'] = game_info['genres']
        game_data[game_info['name'].lower()] = new_game_info
    return game_data

max_item_id = 17408
model_name = "data/embedding_model/Qwen/Qwen3-Embedding-0.6B"
backend = 'qwen'
num_workers = 2

item2id: Dict[str, str] = load_json(json_path="data/model_configs/item2id.json")

known_item_info = []
known_indices = []
unknown_indices = []
item2embedding_indice = [-1] * (max_item_id)
for item_title, item_id in item2id.items():
    if item_title != f"unknown_item_{item_id}":
        title = delete_platform_info(item_title)
        known_item_info.append(title.lower())
        known_indices.append(int(item_id))
        item2embedding_indice[item_id - 1] = len(known_indices) - 1
    else:
        unknown_indices.append(int(item_id))
        item2embedding_indice[item_id - 1] = -1

embeddings = encode(
    model_path=model_name, 
    texts=known_item_info,
    batch_size=64,
    num_workers=num_workers,
    backend=backend
)

embedding_dim = embeddings.shape[1]

index = faiss.IndexFlatIP(embedding_dim)

# download https://huggingface.co/datasets/FronkonGames/steam-games-dataset/blob/main/games.json
external_game_json = "data/external_dataset/FronkonGames/steam-games-dataset/games.json"
external_games: Dict[int, Any] = load_external_dataset(external_game_json)

external_titles = list(external_games.keys())
external_ids = [external_games[title]['id'] for title in external_titles]

external_embeddings = encode(
    model_path=model_name, 
    texts=external_titles,
    batch_size=64,
    num_workers=num_workers,
    backend=backend
)
external_embeddings = external_embeddings.astype(np.float32)

index = faiss.IndexFlatIP(embedding_dim)
index.add(external_embeddings)

print(f"FAISS index built. Total external items: {len(external_titles)}")

item_embeddings = embeddings.astype(np.float32)
scores, retrieved_idx = index.search(item_embeddings, 1)

matching_results = []
threshold = 0.6

for item_title, item_id in item2id.items():
    item_id = int(item_id)
    data_info = {}
    if (
        (item2embedding_indice[item_id - 1] != -1 and scores[item2embedding_indice[item_id - 1]][0] >= threshold)
        or
        delete_platform_info(item_title).lower() in external_titles
    ):
        ext_game_info = external_games[external_titles[retrieved_idx[item2embedding_indice[item_id - 1]][0]]]
        data_info["item_id"] = item_id
        data_info["item_title"] = item_title
        data_info["score"] = scores[item2embedding_indice[item_id - 1]][0].item()
        data_info["game_title"] = ext_game_info["title"]
        data_info["release_year"] = ext_game_info["release_year"]
        data_info["price"] = ext_game_info["price"]
        data_info["description"] = ext_game_info["description"]
        data_info["developers"] = ext_game_info["developers"]
        data_info["genres"] = ext_game_info["genres"]
    else:
        data_info["item_id"] = item_id
        data_info["item_title"] = item_title
        data_info["score"] = (
            scores[item2embedding_indice[item_id - 1]][0].item()
            if item2embedding_indice[item_id - 1] != -1 else 0.0
        )
        data_info["game_title"] = "Unknown"
        data_info["release_year"] = "Unknown"
        data_info["price"] = "Unknown"
        data_info["description"] = "Unknown"
        data_info["developers"] = ["Unknown"]
        data_info["genres"] = ["Unknown"]
    matching_results.append(data_info)

save_jsonl(matching_results, save_path="data_preprocess/build_pretrained_embeddings/outputs/steam_information.jsonl")