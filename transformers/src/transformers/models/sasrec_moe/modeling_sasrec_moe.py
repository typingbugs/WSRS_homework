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
"""PyTorch SASRec-MOE model."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "SasrecMoEPreTrainedModel",
    "SasrecMoEModel",
    "SasrecMoEForCausalLM",
]

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast, MoeCausalLMOutputWithPast
def switch_load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """
    Switch Transformer style load balancing loss - encourages but doesn't force uniform distribution.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    # Switch Transformer style: use coefficient of variation instead of forcing uniform
    # This allows some experts to be used more than others
    mean_tokens = torch.mean(tokens_per_expert)
    std_tokens = torch.std(tokens_per_expert)
    cv_tokens = std_tokens / (mean_tokens + 1e-10)  # Coefficient of variation
    
    # Also penalize very low utilization
    min_utilization_penalty = torch.relu(0.1 - tokens_per_expert).sum()
    
    return cv_tokens + 0.1 * min_utilization_penalty
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_sasrec_moe import SasrecMoEConfig
from ...integrations import use_kernel_forward_from_hub

logger = logging.get_logger(__name__)


class SasrecMoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SasrecTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        
        # Initialize weights properly
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        
        # Only routed experts, no shared experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.moe_intermediate_size),
                ACT2FN[config.hidden_act],
                nn.Dropout(config.moe_dropout),
                nn.Linear(config.moe_intermediate_size, config.hidden_size),
            ) for _ in range(config.n_routed_experts)
        ])
        
        # Switch Transformer style router
        self.gate = SasrecTopkRouter(config)
        
        # Switch Transformer style routing parameters
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.capacity_factor = config.capacity_factor
        
        # Calculate expert capacity
        if config.expert_capacity is not None:
            self.expert_capacity = config.expert_capacity
        else:
            # Auto-calculate capacity based on Switch Transformer paper
            self.expert_capacity = None  # Will be calculated per batch
        
        self.dropout = nn.Dropout(config.moe_dropout)
        
        self.dropout = nn.Dropout(config.moe_dropout)

    def route_tokens_to_experts(self, router_logits):
        # Add noise for exploration (Switch Transformer style)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
        
        # Use softmax instead of sigmoid
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Top-k routing (keep top-2)
        topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_probs.gather(1, topk_indices)
        
        # Normalize top-k weights
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
            
        # Scale down expert outputs (divide by topk)
        topk_weights = topk_weights / self.top_k
        
        return topk_indices, topk_weights

    def forward(self, x):
        # x: (batch, seq, hidden)
        orig_shape = x.shape
        
        # Remove shared experts - only use routed experts
        # Routed experts with capacity and overflow drop
        router_logits = self.gate(x)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        # Apply capacity and overflow drop (Switch Transformer style)
        batch_size, seq_len = x.shape[:2]
        if self.expert_capacity is None:
            # Auto-calculate capacity: tokens_per_expert = (batch_size * seq_len * top_k) / n_experts
            capacity = int(batch_size * seq_len * self.top_k * self.capacity_factor / self.n_routed_experts)
        else:
            capacity = self.expert_capacity
        
        # Count tokens per expert
        expert_counts = torch.zeros(self.n_routed_experts, dtype=torch.long, device=x.device)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_counts.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=torch.long))
        
        # Apply overflow drop
        overflow_mask = torch.zeros_like(topk_indices, dtype=torch.bool)
        for expert_id in range(self.n_routed_experts):
            expert_mask = (topk_indices == expert_id)
            if expert_counts[expert_id] > capacity:
                # Randomly drop excess tokens
                expert_positions = expert_mask.nonzero(as_tuple=True)[0]
                num_to_drop = expert_counts[expert_id] - capacity
                drop_indices = expert_positions[torch.randperm(len(expert_positions))[:num_to_drop]]
                overflow_mask[drop_indices, (topk_indices[drop_indices] == expert_id).nonzero(as_tuple=True)[1]] = True
        
        # Zero out weights for dropped tokens
        topk_weights = topk_weights.masked_fill(overflow_mask, 0.0)
        
        # Apply routed experts
        x_flat = x.view(-1, x.shape[-1])
        routed_out = torch.zeros_like(x_flat)
        
        # For each expert, accumulate contributions
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_weights = topk_weights[:, i]
            
            for j, expert in enumerate(self.routed_experts):
                mask = (expert_idx == j) & (~overflow_mask[:, i])
                if mask.sum() > 0:
                    expert_input = x_flat[mask]
                    expert_output = expert(expert_input)
                    # Weight and add to routed_out
                    weighted_output = expert_output * expert_weights[mask].unsqueeze(-1)
                    routed_out[mask] += weighted_output
        
        routed_out = routed_out.view(orig_shape)
        
        return self.dropout(routed_out), router_logits


class SasrecMoEAttention(nn.Module):
    def __init__(self, config: SasrecMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = SasrecMoERMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        output_attentions: bool,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, tgt_len, _ = hidden_states.size()
        query = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value = self._shape(self.v_proj(hidden_states), tgt_len, bsz)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = torch.cat([past_k, key], dim=2)
            value = torch.cat([past_v, value], dim=2)
        present = (key, value)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1)
        attn_output = self.o_proj(attn_output)
        hidden_states = self.norm(hidden_states + self.dropout(attn_output))
        attn_probs = attn_weights if output_attentions else None
        return hidden_states, attn_probs, present


class SasrecMoEBlock(nn.Module):
    def __init__(self, config: SasrecMoEConfig):
        super().__init__()
        self.attn = SasrecMoEAttention(config)
        self.moe = MoE(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        output_attentions: bool,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        hidden_states, attn_weights, present_kv = self.attn(
            hidden_states,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        moe_out, gate_logits = self.moe(hidden_states)
        hidden_states = self.norm(hidden_states + self.dropout(moe_out))
        return hidden_states, attn_weights, present_kv, gate_logits
    

class SasrecMoEPreTrainedModel(PreTrainedModel):
    config_class = SasrecMoEConfig
    base_model_prefix = "sasrec_moe"
    supports_gradient_checkpointing = False
    _no_split_modules = ["SasrecMoEBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

class SasrecMoEModel(SasrecMoEPreTrainedModel):
    def __init__(self, config: SasrecMoEConfig):
        super().__init__(config)
        self.item_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([SasrecMoEBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.item_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.item_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, 'output_attentions', False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, 'output_hidden_states', False)
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'return_dict', True)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length, _ = inputs_embeds.shape

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values = past_key_values if past_key_values is not None else tuple([None] * len(self.layers))
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must have the same length as the number of layers")

        if inputs_embeds is None:
            inputs_embeds = self.item_embeddings(input_ids)

        if position_ids is None:
            past_length = past_key_values[0][0].size(2) if past_key_values[0] is not None else 0
            position_ids = torch.arange(past_length, past_length + seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.clamp(0, self.config.max_position_embeddings - 1)

        pos_embeddings = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + pos_embeddings
        hidden_states = self.dropout(hidden_states)

        if attention_mask is None:
            if input_ids is not None and self.config.pad_token_id is not None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
            else:
                attention_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.long)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        present_key_values = () if use_cache else None

        past_length = past_key_values[0][0].size(2) if past_key_values[0] is not None else 0

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past = past_key_values[layer_idx]

            kv_past_len = past[0].size(2) if past is not None else 0
            total_k_len = kv_past_len + seq_length

            if attention_mask is not None:
                past_padding = torch.ones((batch_size, kv_past_len), device=device, dtype=attention_mask.dtype)
                attention_mask_layer = torch.cat([past_padding, attention_mask], dim=1)
                attn_bias = (1.0 - attention_mask_layer[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
            else:
                attn_bias = None

            causal_mask = torch.full((seq_length, total_k_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if attn_bias is not None:
                attn_mask = causal_mask + attn_bias
            else:
                attn_mask = causal_mask

            hidden_states, attn_weights, present, gate_logits = layer(
                hidden_states,
                attn_mask=attn_mask,
                output_attentions=output_attentions,
                past_key_value=past,
            )
            if output_router_logits:
                all_router_logits = all_router_logits + (gate_logits,)
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
            if use_cache:
                present_key_values = present_key_values + (present,)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_values, all_hidden_states, all_attentions, all_router_logits] if v is not None)

        if use_cache:
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=present_key_values,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
            if output_router_logits:
                output.router_logits = all_router_logits
            return output

        output = BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)
        if self.config.output_router_logits:
            output.router_logits = all_router_logits
        return output

class SasrecMoEForCausalLM(SasrecMoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "sasrec_moe.item_embeddings.weight"}
    supports_cache = True

    def __init__(self, config: SasrecMoEConfig):
        super().__init__(config)
        self.sasrec_moe = SasrecMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.n_routed_experts = config.n_routed_experts
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.sasrec_moe.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.sasrec_moe.set_input_embeddings(new_embeddings)
        self.lm_head.weight = new_embeddings.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs,
    ):
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        outputs = self.sasrec_moe(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else getattr(self.config, 'return_dict', True),
            output_router_logits=output_router_logits,
        )
        resolved_return_dict = return_dict if return_dict is not None else getattr(self.config, 'return_dict', True)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        aux_loss = None
        if output_router_logits:
            aux_loss = switch_load_balancing_loss_func(
                outputs.router_logits,
                self.n_routed_experts,
                self.config.num_experts_per_tok,  # Use actual top-k from config
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
        if not resolved_return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        if isinstance(outputs, BaseModelOutputWithPast):
            moe_output = MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=outputs.router_logits if output_router_logits else None,
            )
            return moe_output
        # For cases without past_key_values, create a custom output with aux_loss
        output = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=None,
        )
        output.aux_loss = aux_loss
        if output_router_logits:
            output.router_logits = outputs.router_logits
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -past_key_values[0][0].size(2) - 1 :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values