import torch
import numpy as np


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [batch, vocab_size]
    labels: [batch]
    返回: top-10 索引
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    # 将 pad/token 范围置为 -inf，避免被选中
    logits[:, 0] = -float("inf")       # pad_token
    logits[:, 17409:] = -float("inf")  # vocab 尾部

    top10_values, top10_indices = torch.topk(logits, k=10, dim=-1)
    return top10_indices.contiguous()


def compute_metrics(eval_pred):
    """
    eval_pred.predictions: [batch, 10]  # top-10 indices
    eval_pred.label_ids: [batch]
    """
    top10_indices = eval_pred.predictions
    if isinstance(top10_indices, torch.Tensor):
        top10_indices = top10_indices.cpu().numpy()
    
    labels = eval_pred.label_ids
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    rr = []
    for t10, gt in zip(top10_indices, labels):
        if gt in t10:
            rank = np.where(t10 == gt)[0][0] + 1
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)

    return {"mrr@10": float(np.mean(rr))}
