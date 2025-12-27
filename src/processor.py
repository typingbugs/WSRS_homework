import json
import os
from typing import List, Dict, Any, Optional
import torch
from transformers.processing_utils import ProcessorMixin


class RecommendProcessor(ProcessorMixin):
    # ProcessorMixin expects attributes to be processor components (like tokenizer, image_processor)
    # For our use case, we directly store item2id as a dict
    attributes = []
    
    def __init__(
        self,
        item2id: Dict[str, int] = None,
        max_seq_len: int = 768,
        pad_token_id: int = 0,
        unk_token_id: int = 17409,
        bos_token_id: int = 17410,
        eos_token_id: int = 17411,
        **kwargs
    ):
        self.item2id = item2id or {}
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from a directory containing preprocessor_config.json and item2id.json
        
        Args:
            pretrained_model_name_or_path: Path to the directory containing processor files
            **kwargs: Additional keyword arguments
            
        Returns:
            RecommendProcessor instance
        """
        config_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # load item2id mapping
        item2id_path = os.path.join(pretrained_model_name_or_path, "item2id.json")
        with open(item2id_path, "r", encoding="utf-8") as f:
            item2id = json.load(f)

        # Remove 'processor_class' from config if present
        config.pop("processor_class", None)
        
        # Merge with kwargs
        config.update(kwargs)
        
        return cls(item2id=item2id, **config)

    def encode_titles(self, titles: List[str], add_special_tokens: bool = True) -> List[int]:
        """
        Convert a list of item_titles to list of item_ids.
        Unknown titles get unk_token_id.
        """
        item_ids = [self.bos_token_id] if add_special_tokens else []
        for t in titles:
            if t in self.item2id:
                item_ids.append(self.item2id[t])
            else:
                item_ids.append(self.unk_token_id)
        return item_ids
    
    def decode_titles(self, item_ids: List[int], skip_special_tokens: bool = True) -> List[str]:
        """
        Convert a list of item_ids back to item_titles.
        Unknown ids get 'unknown_item_{id}'.
        """
        id2item = {v: k for k, v in self.item2id.items()}
        titles = []
        for i in item_ids:
            if skip_special_tokens and i in {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}:
                continue
            titles.append(id2item.get(i, f"unknown_item_{i}"))
        return titles

    def __call__(
        self,
        histories: List[List[str]] | List[List[int]],
        padding: bool = True,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        
        batch_item_ids = []
        batch_attention_masks = []

        for history in histories:
            if not isinstance(history, list):
                raise ValueError("Each history should be a list of item titles.")
            if isinstance(history[0], str):
                item_ids = self.encode_titles(history, add_special_tokens=add_special_tokens)
            else:
                if add_special_tokens:
                    item_ids = [self.bos_token_id] + history.copy()
                else:
                    item_ids = history.copy()

            # 2) Trim if too long
            if len(item_ids) > self.max_seq_len:
                item_ids = item_ids[-self.max_seq_len:]

            # 3) Pad sequence
            if padding:
                pad_len = self.max_seq_len - len(item_ids)
                if pad_len > 0:
                    item_ids = item_ids + [self.pad_token_id] * pad_len

            attention_mask = [1 if x != self.pad_token_id else 0 for x in item_ids]

            batch_item_ids.append(item_ids)
            batch_attention_masks.append(attention_mask)

        # 4) Convert to output dict
        data = {
            "input_ids": batch_item_ids,
            "attention_mask": batch_attention_masks,
        }

        # 5) Convert to tensors
        if return_tensors == "pt":
            data = {k: torch.tensor(v) for k, v in data.items()}  # [batch_size, seq_len]

        return data

    def save_pretrained(self, save_directory: str):
        """
        Save processor configuration and mappings to a directory
        
        Args:
            save_directory: Path to the directory where files will be saved
        """
        os.makedirs(save_directory, exist_ok=True)

        # write config
        config = {
            "processor_class": "RecommendProcessor",
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # write item2id
        item2id_path = os.path.join(save_directory, "item2id.json")
        with open(item2id_path, "w", encoding="utf-8") as f:
            json.dump(self.item2id, f, ensure_ascii=False, indent=2)
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = {
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        output["processor_class"] = self.__class__.__name__
        return output
    
    @property
    def model_input_names(self):
        """
        Returns the list of model input names.
        """
        return ["input_ids", "attention_mask"]
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_seq_len={self.max_seq_len}, "
            f"vocab_size={len(self.item2id) + 4}, "
            f"pad_token_id={self.pad_token_id})"
        )
