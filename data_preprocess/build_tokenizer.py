import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast

def build_hf_tokenizer(item2id: dict, save_dir: str, max_length: int = 20):
    """
    将 item2id 转成 Hugging Face tokenizer 文件夹

    Args:
        item2id (dict): {"item_name": id, ...}
        save_dir (str): 保存 tokenizer 的目录
        max_length (int): tokenizer 最大序列长度
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1️⃣ 增加特殊 token
    special_tokens = {
        "<PAD>": 0,
        "<UNK>": len(item2id) + 1,
        "<BOS>": len(item2id) + 2,
        "<EOS>": len(item2id) + 3
    }
    
    vocab = {**item2id, **special_tokens}
    
    # 2️⃣ 保存 vocab.json
    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 3️⃣ 创建 tokenizer.json (WordLevel)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
    Regex(r"(<[^>]+>)"),
    behavior="isolated"
)
    
    # 添加 post_processor 支持 BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> $B:1 <EOS>:1",
        special_tokens=[("<BOS>", vocab["<BOS>"]), ("<EOS>", vocab["<EOS>"])]
    )
    
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # 4️⃣ 保存 tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": max_length,
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "unk_token": "<UNK>"
    }
    
    config_path = os.path.join(save_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    print(f"Tokenizer saved to {save_dir}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Special tokens: {special_tokens}")

    # 5️⃣ 测试加载
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    print("Example encoding:", hf_tokenizer.encode("Super Mario 64 Diddy Kong Racing"))


def load_item2id(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    item2id = {f"<{k}>": int(v) for k, v in item2id.items()}
    return item2id

# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    item2id = load_item2id("data_preprocess/outputs/item2id.json")
    
    save_dir = "data_preprocess/outputs/hf_item_tokenizer"
    build_hf_tokenizer(item2id, save_dir)
