# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SASRec configuration."""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

__all__ = ["SasrecGateConfig"]

logger = logging.get_logger(__name__)


class SasrecGateConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SasrecGateModel`].

    It is based on the SASRec sequential recommendation architecture and is registered with
    [`~transformers.AutoConfig`] and [`~transformers.AutoModelForCausalLM`].

    Args:
        vocab_size (`int`, *optional*, defaults to 30000): Vocabulary size (number of items) of the model.
        hidden_size (`int`, *optional*, defaults to 128): Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 2): Number of Transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 4): Number of attention heads.
        max_position_embeddings (`int`, *optional*, defaults to 200): Maximum sequence length that this model might ever be used with.
        hidden_dropout (`float`, *optional*, defaults to 0.1): Dropout probability for embeddings and fully connected layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1): Dropout probability applied on attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02): Standard deviation for initializing weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12): Epsilon for layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0): Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1): Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2): End of sequence token id.
        use_cache (`bool`, *optional*, defaults to `False`): Whether to enable key-value caching. SASRec does not implement caching by default.
    """

    model_type = "sasrec_gate"

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 128,
        intermediate_size: int = 512,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        max_position_embeddings: int = 200,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "gelu",
        use_cache: bool = False,
        use_return_dict: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            return_dict=use_return_dict,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        # SASRec uses item ids as tokens; tie embeddings by default.
        self.tie_word_embeddings = True