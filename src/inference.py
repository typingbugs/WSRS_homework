"""
Inference script for next item prediction.
Loads a trained model checkpoint and performs next item prediction.
"""
import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM
from src.processor import RecommendProcessor


class ItemPredictor:
    def __init__(self, model_path, processor_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self.model.eval()
        self.processor = RecommendProcessor.from_pretrained(processor_path)
    
    def predict_next_items_batch(self, batch_history_item_ids, top_k=10):
        encoded = self.processor(
            batch_history_item_ids,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Move to the same device as model
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Get logits for the last valid position (before padding)
        last_valid_pos = attention_mask.sum(dim=1) - 1  # [batch_size]
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=logits.device)
        last_logits = logits[batch_indices, last_valid_pos, :]  # [batch_size, vocab_size]

        last_logits[:, 0] = -float("inf")
        last_logits[:, 17409:] = -float("inf")
        
        # Get top-k predictions
        top_k_logits, top_k_ids = torch.topk(last_logits, k=min(top_k, last_logits.shape[-1]), dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)  # [batch_size, top_k]
        
        top_k_ids = top_k_ids.cpu().numpy().tolist()
        top_k_probs = top_k_probs.cpu().numpy().tolist()
        top_k_names = [self.processor.decode_titles(ids, skip_special_tokens=True) for ids in top_k_ids]

        return top_k_ids, top_k_names, top_k_probs
    

def predict_from_file(predictor: ItemPredictor, test_file: str, output_path: str, top_k: int = 10, batch_size: int = 32):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]

    results = []
    for batch in tqdm(batches, desc="Predicting next items"):
        batch_user_id = [example['user_id'] for example in batch]
        batch_histories = [example['title_sequence'] for example in batch]

        top_k_ids, top_k_names, top_k_probs = predictor.predict_next_items_batch(
            batch_histories, top_k=top_k
        )

        for user_id, top_k_ids in zip(batch_user_id, top_k_ids):
            result = {
                "user_id": user_id,
                "item_id": top_k_ids,
            }
            results.append(result)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    return results


def get_args():
    parser = argparse.ArgumentParser(description="Next item prediction inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint directory")
    parser.add_argument("--processor_path", type=str, default="data/model_configs", help="Path to the processor config directory")
    parser.add_argument("--test_file", type=str, default="data/cleaned_data/test_dataset.jsonl", help="Path to test dataset JSONL file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top predictions to return")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results (optional)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    predictor = ItemPredictor(model_path=args.model_path, processor_path=args.processor_path)
    predict_from_file(predictor=predictor, test_file=args.test_file, output_path=args.output_file, top_k=args.top_k, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
