from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
import yaml
import argparse
from typing import Literal

@dataclass
class DataArguments:
    data_dir: str = field(
        default="data/cleaned_data",
        metadata={"help": "Directory containing the dataset files."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="data/model_configs",
        metadata={"help": "model path."}
    )
    processor_name_or_path: str = field(
        default="data/model_configs",
        metadata={"help": "Path to the processor config."}
    )
    embedding_path: str = field(
        default="data/item_embeddings/all-mpnet-base-v2.npy",
        metadata={"help": "Path to load initial item embeddings."}
    )
    stage: Literal[1, 2] = field(
        default=1,
        metadata={"help": "Training stage: 1 or 2."}
    )
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    training_parser = HfArgumentParser((
        TrainingArguments,
        DataArguments,
        ModelArguments
    ))
    training_args, data_args, model_args = training_parser.parse_dict(config)
    return training_args, data_args, model_args