from dataclasses import dataclass, field
from transformers import HfArgumentParser
from trl import DPOConfig
import yaml
import argparse

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
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    training_parser = HfArgumentParser((
        DPOConfig,
        DataArguments,
        ModelArguments
    ))
    training_args, data_args, model_args = training_parser.parse_dict(config)
    return training_args, data_args, model_args