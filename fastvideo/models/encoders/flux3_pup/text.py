# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modified for FastVideo native inference.
"""Qwen3.5 hybrid text backbone with native recurrent and KV caches."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from fastvideo.layers.linear import ReplicatedLinear


class Linear(ReplicatedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


@dataclass
class CacheLayer:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    conv_states: torch.Tensor | None = None
    recurrent_states: torch.Tensor | None = None


class HybridCache:
    def __init__(self, config):
        self.layers = [CacheLayer() for _ in range(config.num_hidden_layers)]
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        if layer_idx is None:
            return any(layer.conv_states is not None for layer in self.layers)
        return self.layers[layer_idx].conv_states is not None

    def update(self, keys, values, layer_idx):
        layer = self.layers[layer_idx]
        layer.keys = keys if layer.keys is None else torch.cat((layer.keys, keys), dim=-2)
        layer.values = values if layer.values is None else torch.cat((layer.values, values), dim=-2)
        return layer.keys, layer.values

    def update_conv_state(self, states, layer_idx):
        layer = self.layers[layer_idx]
        if layer.conv_states is None:
            layer.conv_states = states.clone()
        else:
            layer.conv_states.copy_(states[..., -layer.conv_states.shape[-1]:])
        return layer.conv_states

    def update_recurrent_state(self, states, layer_idx):
        layer = self.layers[layer_idx]
        if layer.recurrent_states is None:
            layer.recurrent_states = torch.empty_like(states, dtype=layer.conv_states.dtype)
        layer.recurrent_states.copy_(states)
        return layer.recurrent_states


def sdpa_attention_forward(module, query, key, value, attention_mask):
    if attention_mask is not None:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask,
        dropout_p=module.attention_dropout if module.training else 0.0,
        scale=module.scaling, is_causal=query.shape[2] > 1 and attention_mask is None,
        enable_gqa=attention_mask is None,
    )
    return output.transpose(1, 2).contiguous(), None


class TextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope = config.rope_parameters
        if rope.get("rope_type", "default") != "default":
            raise ValueError("FLUX.3 PUP requires default RoPE")
        self.theta = rope["rope_theta"]
        self.dim = int(config.head_dim * rope.get("partial_rotary_factor", 1.0))
        self.mrope_section = rope.get("mrope_section", [11, 11, 10])
        self.register_buffer("inv_freq", self._inv_freq(torch.empty(0).device), persistent=False)

    def _inv_freq(self, device=None):
        frequencies = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device="cpu").float() / self.dim))
        return frequencies.to(device=device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)
        inv = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv @ position_ids[:, :, None, :].float()).transpose(2, 3)
            freqs_t = freqs[0]
            for dim, offset in enumerate((1, 2), start=1):
                index = slice(offset, self.mrope_section[dim] * 3, 3)
                freqs_t[..., index] = freqs[dim, ..., index]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)



class Qwen3_5RMSNormGated(nn.Module):

    def __init__(self, hidden_size, eps=1e-06, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1 and (attention_mask.shape[0] > 1):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    (_, hidden_size, seq_len) = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int=-1, eps: float=1e-06):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                output_final_state=False, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-06)
        key = l2norm(key, dim=-1, eps=1e-06)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    (batch_size, num_heads, sequence_length, k_head_dim) = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()
    attn = -(k_beta @ key.transpose(-1, -2) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        (q_i, k_i, v_i) = (query[:, :, i], key[:, :, i], value[:, :, i])
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = q_i * g[:, :, i, :, None].exp() @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return (core_attn_out, last_recurrent_state)


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state,
                                    use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-06)
        key = l2norm(key, dim=-1, eps=1e-06)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    (batch_size, num_heads, sequence_length, k_head_dim) = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim,
                               dtype=value.dtype, device=value.device)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return (core_attn_out, last_recurrent_state)


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = F.silu
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
                                kernel_size=self.conv_kernel_size, groups=self.conv_dim,
                                padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)
        self.causal_conv1d_update = torch_causal_conv1d_update
        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        self.in_proj_qkv = Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor, cache_params: HybridCache | None = None,
                attention_mask: torch.Tensor | None = None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        (batch_size, seq_len, _) = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        if use_precomputed_states and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1),
                                                 self.conv1d.bias, self.activation)
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        (query, key, value) = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    (q_rot, q_pass) = (q[..., :rotary_dim], q[..., rotary_dim:])
    (k_rot, k_pass) = (k[..., :rotary_dim], k[..., rotary_dim:])
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return (q_embed, k_embed)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    (batch, num_key_value_heads, slen, head_dim) = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3_5Attention(nn.Module):

    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2,
                             bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None, past_key_values: HybridCache | None = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            (key_states, value_states) = past_key_values.update(key_states, value_states, self.layer_idx)
        attn_output, attn_weights = sdpa_attention_forward(
            self, query_states, key_states, value_states, attention_mask
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class Qwen3_5MLP(nn.Module):

    def __init__(self, config: SimpleNamespace, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3_5RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.eps}'


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == 'full_attention':
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None,
                past_key_values: HybridCache | None = None, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == 'linear_attention':
            hidden_states = self.linear_attn(hidden_states=hidden_states, cache_params=past_key_values,
                                             attention_mask=attention_mask)
        elif self.layer_type == 'full_attention':
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, position_embeddings=position_embeddings, **kwargs,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@dataclass
class TextModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: HybridCache | None = None


def _causal_mask(inputs, attention_mask, past_length, text_positions=None, detect_packing=False):
    batch, length = inputs.shape[:2]
    total = past_length + length
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask
    packed = None
    if detect_packing and text_positions is not None and attention_mask is None:
        diff = torch.diff(text_positions, prepend=text_positions[:, :1] - 1, dim=-1)
        indices = (diff != 1).cumsum(-1)
        if not bool((indices[:, -1] == 0).all()):
            packed = indices
    if packed is None and (length == 1 or past_length == 0):
        if attention_mask is None or bool(attention_mask.all()):
            return None
    queries = torch.arange(length, device=inputs.device) + past_length
    keys = torch.arange(total, device=inputs.device)
    mask = (keys[None, :] <= queries[:, None])[None, None].expand(batch, 1, length, total)
    if attention_mask is not None:
        mask = mask & attention_mask[:, None, None, :total].to(device=inputs.device, dtype=torch.bool)
    if packed is not None:
        mask = mask & (packed[:, None, :, None] == packed[:, None, None, :])
    return mask


class Flux3PupTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        if config.hidden_act != "silu":
            raise ValueError("FLUX.3 PUP requires SiLU activation")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def materialize_non_persistent_buffers(self, device, dtype=None):
        self.rotary_emb.inv_freq = self.rotary_emb._inv_freq(device)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = HybridCache(self.config)
        batch, length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            positions = torch.arange(length, device=inputs_embeds.device) + past_length
            position_ids = positions.view(1, 1, -1).expand(4, batch, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(4, position_ids.shape[0], -1)
        text_positions = None
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_positions = position_ids[0]
            position_ids = position_ids[1:]
        causal_mask = _causal_mask(inputs_embeds, attention_mask, past_length, text_positions,
                                   detect_packing=past_key_values is None)
        linear_mask = attention_mask
        if ((past_key_values is not None and past_key_values.has_previous_state())
                or (attention_mask is not None and bool((attention_mask == 1).all()))):
            linear_mask = None
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers):
            mask = linear_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=mask,
                                  position_ids=text_positions, past_key_values=past_key_values)
        hidden_states = self.norm(hidden_states)
        if past_key_values is not None:
            past_key_values.seen_tokens += length
        return TextModelOutput(hidden_states, past_key_values)


TextModel = Flux3PupTextModel
