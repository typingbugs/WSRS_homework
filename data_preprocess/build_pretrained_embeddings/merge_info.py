import os
from typing import Dict, Any, List
import json


def load_jsonl(jsonl_path: str):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

    
def save_jsonl(data: List[Dict[str, Any]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_str_info(info_1: str, info_2: str) -> str:
    if info_1 != "Unknown":
        return info_1
    elif info_2 != "Unknown":
        return info_2
    else:
        return "Unknown"
    
def get_list_info(info_1: list[str], info_2: list[str]) -> list[str]:
    res_set = set(["Unknown"])
    res_set.update(info_1)
    res_set.update(info_2)
    if len(res_set) > 1:
        res_set.remove("Unknown")
    return list(res_set)


def merge_item_information():
    steam_info = load_jsonl("data_preprocess/build_pretrained_embeddings/outputs/steam_information.jsonl")
    wiki_info = load_jsonl("data_preprocess/build_pretrained_embeddings/outputs/wiki_information.jsonl")

    wiki_info_dict = {item['item_id']: item for item in wiki_info}
    steam_info_dict = {item['item_id']: item for item in steam_info}

    merged_info = []
    for item_id in wiki_info_dict.keys():
        item_steam_info = steam_info_dict[item_id]
        item_wiki_info = wiki_info_dict[item_id]

        item_info = {
            "item_id": item_id,
            "item_title": item_steam_info["item_title"]
        }
        # release_year
        item_info["release_year"] = get_str_info(item_steam_info["release_year"], item_wiki_info.get("release_year", "Unknown"))
        item_info["price"] = get_str_info(str(item_steam_info["price"]), str(item_wiki_info.get("price", "Unknown")))
        item_info["developers"] = get_list_info(item_steam_info["developers"], item_wiki_info.get("developers", ["Unknown"]))
        item_info["genres"] = get_list_info(item_steam_info["genres"], item_wiki_info.get("genres", ["Unknown"]))
        item_info["item_type"] = get_str_info("Game" if item_steam_info["game_title"] != "Unknown" else "Unknown", item_wiki_info.get("item_type", "Unknown"))
        item_info["platform"] = get_list_info(["PC"] if item_steam_info["game_title"] != "Unknown" else ["Unknown"], item_wiki_info.get("platform", ["Unknown"]))
        item_info["info"] = "Item Title: {}; Item Type: {}; Release Year: {}; Genres: {}; Platform: {}; Price: {}; Developers: {}".format(
            item_info["item_title"],
            item_info["item_type"],
            item_info["release_year"],
            ", ".join(item_info["genres"]),
            ", ".join(item_info["platform"]),
            item_info["price"] + (" USD" if item_info["price"] != "Unknown" else ""),
            ", ".join(item_info["developers"])
        )
        merged_info.append(item_info)

    save_jsonl(merged_info, "data_preprocess/build_pretrained_embeddings/outputs/merged_information.jsonl")
        

if __name__ == "__main__":
    merge_item_information()