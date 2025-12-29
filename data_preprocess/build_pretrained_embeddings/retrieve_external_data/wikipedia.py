import wikipediaapi
import json
import os
from typing import Dict, Any
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv


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

def get_wikipedia_summary(wiki_wiki, title: str) -> str:
    max_retries = 3
    try:
        while max_retries > 0:
            page = wiki_wiki.page(title)
            if page.exists():
                return page.summary
            else:
                return "No summary available."
    except Exception as e:
        max_retries -= 1
        if max_retries == 0:
            print(title)
            return "No summary available."
    

item2id = load_json("data_preprocess/outputs/item2id.json")
load_dotenv()
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent=f'MyGameWikiBot/1.0 ({os.getenv("EMAIL_FOR_WIKI_BOT", "")})',
    language='en'
)

item_summaries = []
for item_title, item_id in tqdm(item2id.items(), total=len(item2id)):
    title = delete_platform_info(item_title)
    page = wiki_wiki.page(title)
    page_summary = get_wikipedia_summary(wiki_wiki, title)
    item_summaries.append({
        "item_id": item_id,
        "item_title": item_title,
        "search_title": title,
        "summary": page_summary
    })

save_jsonl(item_summaries, save_path="data_preprocess/build_pretrained_embeddings/outputs/wiki_summaries.jsonl")