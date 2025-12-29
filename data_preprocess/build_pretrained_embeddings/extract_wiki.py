import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import json
from dotenv import load_dotenv
import os
from typing import Dict, Any, List

def load_jsonl(jsonl_path: str):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

    
def save_jsonl(data: List[Dict[str, Any]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

sys_prompt = """
You are an information extraction model. 
Your task is to infer the properties of a video-game-related product based on (1) its title and (2) optional Wikipedia description text.
Extract the following properties:
1. release_year: The year the product was first released. If information is missing or uncertain, output “Unknown”.  
2. price: The retail price of the product at launch in USD. If the price is not available, output "Unknown".
3. genres: The genre of the product. For video games, common genres include Action, Adventure, RPG, Strategy, Simulation, Sports, Racing, Indie, Casual etc. For hardware, output ["Unknown"].
4. developers: The developer(s) of the product. If multiple developers, separate them with commas. If not available, output ["Unknown"].
5. item_type: Either "Game" for video games or "Hardware" for gaming-related hardware products (e.g., consoles, controllers, VR headsets, etc.).
6. platform: The platform(s) the product is available on. For video games, common platforms include PC, PlayStation, Xbox, Nintendo, etc. For hardware, output the platform it is designed for (e.g., ["PlayStation"] for a PS5 controller). If multiple platforms are supported, output them as a list. If the platform is not specified or unclear, output ["Unknown"].
Always produce the output in the exact structured format shown in the examples.

===== FEW-SHOT EXAMPLES =====

[EXAMPLE 1]
Input Title:
"Tomb Raider II"

Input Wikipedia Text:
"Tomb Raider II (also known as Tomb Raider II: Starring Lara Croft) is a 1997 action-adventure video game developed by Core Design and published by Eidos Interactive. It was first released on Windows and PlayStation. Later releases came for Mac OS (1998), iOS (2014) and Android (2015). It is the second entry in the Tomb Raider series, and follows archaeologist-adventurer Lara Croft hunting the magical Dagger of Xian in competition with an Italian cult. Gameplay features Lara navigating levels split into multiple areas and room complexes while fighting enemies and solving puzzles to progress, with some areas allowing for or requiring the use of vehicles.\nProduction began in 1996 immediately after the success of the original Tomb Raider, being completed in between six and eight months, a short development period which was physically and emotionally stressful for the team. Original staff members Toby Gard and Paul Douglas left over creative differences with the publisher, though many remained including composer Nathan McCree. A Sega Saturn version was scrapped due to a console exclusivity deal signed between Eidos and Sony.\nThe game was well-received by critics upon its release, with many noting its expanded gameplay and smoother graphics. It went on to sell nearly seven million copies worldwide. An expansion pack entitled The Further Adventures of Lara Croft was in development in late 1997 but was cancelled. Some elements from the project were carried over to the 1998 sequel, Tomb Raider III. An expansion entitled The Golden Mask was released the following year, containing new levels focused on Lara's quest to find a golden mask in Alaska. A remastered version of the game, alongside The Golden Mask, was included in Tomb Raider I–III Remastered in 2024.

Expected Output:
{"release_year": 1997, "price": "Unknown", "genres": ["Action", "Adventure"], "developers": ["Core Design"], "item_type": "Game", "platform": ["PC", "PlayStation"]}

--------------------------------

[EXAMPLE 2]
Input Title:
"Sony PlayStation 2 DualShock 2 Controller - Black"

Input Wikipedia Text:
"The DualShock 2 is a game controller for the PlayStation 2 video game console, released by Sony."

Expected Output:
{"release_year": "Unknown", "price": "Unknown", "genres": ["Unknown"], "developers": ["Unknown"], "item_type": "Hardware", "platform": ["PlayStation"]}

--------------------------------

[EXAMPLE 3]
Input Title:
"Driver - PC"

Input Wikipedia Text:
"No summary available."

Expected Output:
{"release_year": "Unknown", "price": "Unknown", "genres": ["Racing"], "developers": ["Unknown"], "item_type": "Game", "platform": ["PC"]}

===== END OF FEW-SHOT EXAMPLES =====
""".lstrip()

user_prompt = """
Now process the following item:

Input Title:
"{item_title}"

Input Wikipedia Text:
"{wiki_text}"

Provide output strictly in the same format as in the examples.
""".lstrip()

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))

async def extract_info(example, semaphore: asyncio.Semaphore):
    async with semaphore:   # 控制并发度
        max_retries = 10
        while max_retries > 0:
            try:
                title: str = example["item_title"]
                wiki_text: str = example["summary"]

                if wiki_text is None or len(wiki_text) < 50:
                    wiki_text = "No summary available."

                resp = await client.chat.completions.create(
                    model="qwen/qwen3-32b",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt.format(item_title=title, wiki_text=wiki_text)}
                    ]
                )

                info_str = resp.choices[0].message.content.strip()

                # 检查输出格式是否合法
                try:
                    info = {"item_id": example["item_id"], "title": title}
                    info.update(json.loads(info_str))
                    return info
                except json.JSONDecodeError:
                    raise ValueError(f"Bad format: {info}")

            except Exception as e:
                max_retries -= 1
                if max_retries <= 0:
                    print(e)
                    return {
                        "item_id": example["item_id"], "title": title, "release_year": "Unknown", "price": "Unknown", 
                        "genres": ["Unknown"], "developers": ["Unknown"], "item_type": "Unknown", "platform": ["Unknown"]
                    }
                await asyncio.sleep(0.5)  # backoff

async def main():
    item_summaries = load_jsonl("data_preprocess/build_pretrained_embeddings/outputs/wiki_summaries.jsonl")

    # item_summaries = item_summaries[:20]
    # 控制最大并发数，例如同时跑 10 个（可根据 QPS 调整）
    semaphore = asyncio.Semaphore(8)

    # 异步任务列表
    tasks = [extract_info(item, semaphore) for item in item_summaries]

    # tqdm_asyncio 添加异步进度条
    results = await tqdm_asyncio.gather(*tasks)

    results = [result for result in results]

    save_jsonl(results, "data_preprocess/build_pretrained_embeddings/outputs/wiki_information.jsonl")

if __name__ == "__main__":
    # 在 Notebook 里，替换成 await main()
    asyncio.run(main())