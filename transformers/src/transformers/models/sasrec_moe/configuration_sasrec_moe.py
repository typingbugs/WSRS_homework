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
from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging

__all__ = ["SasrecMoEConfig"]

logger = logging.get_logger(__name__)

class SasrecMoEConfig(PretrainedConfig):
    r"""
    Configuration for SASRec-MoE model.
    """
    model_type = "sasrec_moe"

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 128,
        moe_intermediate_size: int = 512,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        n_routed_experts: int = 4,
        n_shared_experts: int = 0,  # Remove shared experts
        moe_dropout: float = 0.1,
        hidden_act: str = "silu",
        max_position_embeddings: int = 200,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = False,
        use_return_dict: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.01,
        # Switch Transformer style MoE parameters
        num_experts_per_tok: int = 2,  # Top-2 routing
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        # Capacity parameters
        expert_capacity: Optional[int] = None,  # Auto-calculated if None
        capacity_factor: float = 1.0,  # Multiplier for capacity calculation
        **kwargs
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
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_dropout = moe_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        # Switch Transformer style MoE parameters
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        # Capacity parameters
        self.expert_capacity = expert_capacity
        self.capacity_factor = capacity_factor
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
