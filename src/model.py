from transformers import AutoModelForCausalLM, AutoConfig
from src.processor import RecommendProcessor
import numpy as np
import torch
import logging


def init_model(model_args):
    if model_args.stage == 1:
        return init_model_stage1(model_args)
    else:
        return init_model_stage2(model_args)


def init_model_stage1(model_args):
    logger = logging.getLogger(__name__)
    processor = RecommendProcessor.from_pretrained(model_args.processor_name_or_path)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_config(config)

    logger.info(f"Embedding from {model_args.embedding_path}.")
    item_embedding_weights_np = np.load(model_args.embedding_path)
    model_embedding_weights = model.get_input_embeddings().weight
    
    # Get dimensions
    model_dim = model_embedding_weights.shape[1]
    item_embedding_dim = item_embedding_weights_np.shape[1]
    copy_dim = min(model_dim, item_embedding_dim)
    
    logger.info(f"Model embedding dim: {model_dim}, Preprocessed item embedding dim: {item_embedding_dim}, Copy dim: {copy_dim}")
    
    # Convert to tensor and align dimensions
    item_embedding_weights = torch.tensor(
        item_embedding_weights_np[:, :copy_dim], 
        dtype=model_embedding_weights.dtype, 
        device=model_embedding_weights.device
    )
    
    item_range = slice(1, 1 + item_embedding_weights.shape[0])
    with torch.no_grad():
        model_embedding_weights[item_range, :copy_dim].copy_(item_embedding_weights)
    
    # pytorch hook to freeze item embeddings
    def freeze_item_gradients(grad):
        grad[item_range, :copy_dim] = 0
        return grad
    model_embedding_weights.register_hook(freeze_item_gradients)

    return model, processor


def init_model_stage2(model_args):
    processor = RecommendProcessor.from_pretrained(model_args.processor_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    return model, processor
