from transformers import AutoConfig, AutoModelForCausalLM
from src.processor import RecommendProcessor


def init_model(model_args):
    processor = RecommendProcessor.from_pretrained(model_args.processor_name_or_path)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_config(config)

    return model, processor