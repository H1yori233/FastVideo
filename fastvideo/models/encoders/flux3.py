# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo.
"""Text-only Qwen3-VL conditioning for FLUX.3."""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.attention.layer import LocalAttention
from fastvideo.configs.models.encoders.base import BaseEncoderOutput
from fastvideo.configs.models.encoders.flux3 import Flux3TextEncoderConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.models.loader.weight_utils import default_weight_loader
from fastvideo.platforms import AttentionBackendEnum


class Flux3TextRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Flux3TextRotaryEmbedding(nn.Module):

    def __init__(self, config: Flux3TextEncoderConfig):
        super().__init__()
        # CPU initialization preserves the checkpoint reference's RoPE rounding.
        inv_freq = 1.0 / (config.rope_theta **
                          (torch.arange(0, config.head_dim, 2, dtype=torch.int64, device="cpu").float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq.to(torch.empty(0).device), persistent=False)
        self.mrope_section = config.mrope_section

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        positions = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ positions).transpose(2, 3)
            freqs_t = freqs[0]
            for dim, offset in enumerate((1, 2), start=1):
                index = slice(offset, self.mrope_section[dim] * 3, 3)
                freqs_t[..., index] = freqs[dim, ..., index]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Flux3TextAttention(nn.Module):

    def __init__(self, config: Flux3TextEncoderConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.q_proj = ReplicatedLinear(config.hidden_size, config.num_attention_heads * config.head_dim,
                                       bias=config.attention_bias)
        self.k_proj = ReplicatedLinear(config.hidden_size, config.num_key_value_heads * config.head_dim,
                                       bias=config.attention_bias)
        self.v_proj = ReplicatedLinear(config.hidden_size, config.num_key_value_heads * config.head_dim,
                                       bias=config.attention_bias)
        self.o_proj = ReplicatedLinear(config.num_attention_heads * config.head_dim, config.hidden_size,
                                       bias=config.attention_bias)
        self.q_norm = Flux3TextRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Flux3TextRMSNorm(config.head_dim, config.rms_norm_eps)
        self.attn = LocalAttention(config.num_attention_heads, config.head_dim,
                                   num_kv_heads=config.num_key_value_heads, causal=True,
                                   supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
                                   default_backend=AttentionBackendEnum.TORCH_SDPA)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
                has_padding: bool) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states)[0].view(hidden_shape))
        key = self.k_norm(self.k_proj(hidden_states)[0].view(hidden_shape))
        value = self.v_proj(hidden_states)[0].view(hidden_shape)
        cos, sin = (value.unsqueeze(2) for value in position_embeddings)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        if has_padding:
            key = key.repeat_interleave(self.num_key_value_groups, dim=2)
            value = value.repeat_interleave(self.num_key_value_groups, dim=2)
        output = self.attn(query, key, value).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output)[0]


class Flux3TextMLP(nn.Module):

    def __init__(self, config: Flux3TextEncoderConfig):
        super().__init__()
        self.gate_proj = ReplicatedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ReplicatedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = ReplicatedLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)[0]) * self.up_proj(x)[0])[0]


class Flux3TextDecoderLayer(nn.Module):

    def __init__(self, config: Flux3TextEncoderConfig):
        super().__init__()
        self.self_attn = Flux3TextAttention(config)
        self.mlp = Flux3TextMLP(config)
        self.input_layernorm = Flux3TextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Flux3TextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
                has_padding: bool) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_embeddings,
                                                       has_padding)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Flux3TextEncoder(TextEncoder[BaseEncoderOutput]):
    _supported_attention_backends = (AttentionBackendEnum.TORCH_SDPA,)

    def __init__(self, config: Flux3TextEncoderConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Flux3TextDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Flux3TextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Flux3TextRotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, **kwargs: Any) -> BaseEncoderOutput:
        hidden_states = self.embed_tokens(input_ids)
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, batch, length)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        has_padding = not bool(attention_mask.all())
        mask = None
        if has_padding:
            causal = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).tril()
            mask = causal[None, None] & attention_mask[:, None, None, :].bool()
        states = ()
        with set_forward_context(current_timestep=0, attn_metadata=SDPAMetadata(current_timestep=0, attn_mask=mask)):
            for layer in self.layers:
                states += (hidden_states,)
                hidden_states = layer(hidden_states, position_embeddings, has_padding)
        hidden_states = self.norm(hidden_states)
        states += (hidden_states,)
        return BaseEncoderOutput(last_hidden_state=hidden_states, hidden_states=states, attention_mask=attention_mask)

    @torch.inference_mode()
    def encode(self, prompt: str, tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        tokenizer.padding_side = "right"
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                              add_generation_prompt=True)
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.text_len)["input_ids"].shape[1]
        length = min(math.ceil(real / self.config.pad_multiple) * self.config.pad_multiple, self.config.text_len)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        device = self.embed_tokens.weight.device
        output = self(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
        stacked = torch.stack([output.hidden_states[index] for index in self.config.output_layers], dim=1)
        tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
        positions = torch.arange(length, device=device)
        ids = torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)
        return tokens, ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded = set()
        map_name = get_param_names_mapping(self.config.param_names_mapping)
        for name, value in weights:
            # The official multimodal checkpoint includes an unused vision tower.
            if name.startswith("model.visual."):
                continue
            name, _, _ = map_name(name)
            if name not in parameters:
                raise ValueError(f"Unexpected FLUX.3 text encoder tensor: {name}")
            if name in loaded:
                raise ValueError(f"Duplicate FLUX.3 text encoder tensor: {name}")
            parameter = parameters[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, value)
            loaded.add(name)
        missing = parameters.keys() - loaded
        if missing:
            raise ValueError(f"Missing FLUX.3 text encoder tensors: {sorted(missing)}")
        return loaded


EntryClass = Flux3TextEncoder
