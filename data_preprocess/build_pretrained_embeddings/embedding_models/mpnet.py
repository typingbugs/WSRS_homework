from sentence_transformers import SentenceTransformer
import numpy as np
import ray

@ray.remote(num_gpus=1)
class Item2EmbeddingSentenceTransformer:
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(
            model_path, 
            model_kwargs={"attn_implementation": "eager"},
            device='cuda'
        )

    def encode(self, texts: list[str], batch_size: int = 32, truncate_dim: int = 768) -> np.ndarray:
        embeddings = self.model.encode(
            texts, batch_size=batch_size, 
            truncate_dim=truncate_dim,
            show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings