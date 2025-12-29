import ray
from typing import Literal
import numpy as np


def encode(
    model_path: str,
    texts: list[str], 
    batch_size: int = 32,
    truncate_dim: int = 768,
    num_workers: int = 4,
    backend: Literal["bge", "qwen", "mpnet"] = "qwen",
):
    if backend == "bge":
        from .bge import Item2EmbeddingSentenceTransformer as Item2Embedding
    elif backend == "qwen":
        from .qwen import Item2EmbeddingSentenceTransformer as Item2Embedding
    elif backend == "mpnet":
        from .mpnet import Item2EmbeddingSentenceTransformer as Item2Embedding
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    ray.init(num_gpus=num_workers)
    workers = [
        Item2Embedding.options(max_restarts=3).remote(model_path=model_path)
        for _ in range(num_workers)
    ]
    worker_batch_size = (len(texts) + num_workers - 1) // num_workers
    worker_batches = [
        texts[i:i + worker_batch_size]
        for i in range(0, len(texts), worker_batch_size)
    ]
    futures = [
        worker.encode.remote(batch, batch_size=batch_size, truncate_dim=truncate_dim)
        for worker, batch in zip(workers, worker_batches)
    ]
    results = ray.get(futures)
    embeddings = np.vstack(results)
    ray.shutdown()

    return embeddings