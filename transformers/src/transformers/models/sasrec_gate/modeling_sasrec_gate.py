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
"""PyTorch SASRec-Gate model."""

from typing import Optional, Tuple
import torch
from torch import nn

__all__ = [
    "SasrecGatePreTrainedModel",
    "SasrecGateModel",
    "SasrecGateForCausalLM",
]

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_sasrec_gate import SasrecGateConfig
from ...integrations import use_kernel_forward_from_hub

logger = logging.get_logger(__name__)


def _make_causal_mask(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


@use_kernel_forward_from_hub("RMSNorm")
class SasrecGateRMSNorm(nn.Module):
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

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SasrecGateMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SasrecGateAttention(nn.Module):
    def __init__(self, config: SasrecGateConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = SasrecGateRMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        output_attentions: bool,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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


class SasrecGateBlock(nn.Module):
    def __init__(self, config: SasrecGateConfig):
        super().__init__()
        self.attn = SasrecGateAttention(config)
        self.mlp = SasrecGateMLP(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        output_attentions: bool,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        hidden_states, attn_weights, present_kv = self.attn(
            hidden_states,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        ffn_out = self.mlp(hidden_states)
        hidden_states = self.norm(hidden_states + self.dropout(ffn_out))
        return hidden_states, attn_weights, present_kv


class SasrecGatePreTrainedModel(PreTrainedModel):
    config_class = SasrecGateConfig
    base_model_prefix = "sasrec_gate"
    supports_gradient_checkpointing = False
    _no_split_modules = ["SasrecGateBlock"]

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


class SasrecGateModel(SasrecGatePreTrainedModel):
    def __init__(self, config: SasrecGateConfig):
        super().__init__(config)
        self.item_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([SasrecGateBlock(config) for _ in range(config.num_hidden_layers)])
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
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

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

            causal_mask = _make_causal_mask(seq_length, total_k_len, device=hidden_states.device, dtype=hidden_states.dtype)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if attn_bias is not None:
                attn_mask = causal_mask + attn_bias
            else:
                attn_mask = causal_mask

            hidden_states, attn_weights, present = layer(
                hidden_states,
                attn_mask=attn_mask,
                output_attentions=output_attentions,
                past_key_value=past,
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

            if use_cache:
                present_key_values = present_key_values + (present,)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_values, all_hidden_states, all_attentions] if v is not None)

        if use_cache:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=present_key_values,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)


class SasrecGateForCausalLM(SasrecGatePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "sasrec_gate.item_embeddings.weight"}
    supports_cache = True

    def __init__(self, config: SasrecGateConfig):
        super().__init__(config)
        self.sasrec_gate = SasrecGateModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.sasrec_gate.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.sasrec_gate.set_input_embeddings(new_embeddings)
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
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        outputs = self.sasrec_gate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else self.config.return_dict,
        )

        resolved_return_dict = return_dict if return_dict is not None else self.config.return_dict

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not resolved_return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if isinstance(outputs, BaseModelOutputWithPast):
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=None,
        )

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
