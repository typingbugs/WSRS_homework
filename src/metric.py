import torch
import numpy as np


def preprocess_logits_for_metrics(logits, labels):
    """
    在GPU上提取top-10预测，避免传输整个logits到CPU
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    返回: top-10索引和对应的logits值
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    logits[:, :, 0] = -float("inf")
    logits[:, :, 17409:] = -float("inf")
    
    # 只保留top-10的索引和值，大幅减少内存占用
    top10_values, top10_indices = torch.topk(logits, k=10, dim=-1)  # [batch, seq_len, 10]
    
    return top10_indices.contiguous()


def compute_metrics(eval_pred):
    top10_indices = eval_pred.predictions  # [batch, seq_len, 10]
    
    if isinstance(top10_indices, torch.Tensor):
        top10_indices = top10_indices.cpu().numpy()
    
    labels = eval_pred.label_ids  # [batch, seq_len]
    
    reciprocal_ranks_all = []
    reciprocal_ranks_last = []
    
    for pred_seq, label_seq in zip(top10_indices, labels):
        valid_positions = np.where(label_seq != -100)[0]
        
        if len(valid_positions) == 0:
            continue
        
        for pos in valid_positions:
            true_item_id = label_seq[pos]
            top10 = pred_seq[pos]  # shape: [10]
            
            # 检查真实item是否在top-10中
            if true_item_id in top10:
                rank = np.where(top10 == true_item_id)[0][0] + 1  # rank从1开始
                score = 1.0 / rank
            else:
                score = 0.0
            reciprocal_ranks_all.append(score)
            if pos == valid_positions[-1]:  # 仅最后一个位置
                reciprocal_ranks_last.append(score)
    
    mrr_at_10_all = np.mean(reciprocal_ranks_all) if reciprocal_ranks_all else 0.0
    mrr_at_10_last = np.mean(reciprocal_ranks_last) if reciprocal_ranks_last else 0.0
    return {
        "mrr@10_all": mrr_at_10_all,
        "mrr@10_last": mrr_at_10_last
    }