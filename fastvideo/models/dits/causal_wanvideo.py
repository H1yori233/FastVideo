# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
# Wan 1.3B used to require max-autotune for flex attention, but at 768x1280
# teacher-forcing shapes it can spend >15 minutes autotuning per rank. Keep the
# mode runtime-configurable so long ablations do not accidentally pay that cost.
_flex_attention_compile_mode = os.environ.get(
    "FASTVIDEO_FLEX_ATTENTION_COMPILE_MODE", "default").strip()
if _flex_attention_compile_mode.lower() not in {"", "0", "false", "none", "eager"}:
    flex_attention = torch.compile(
        flex_attention, dynamic=False, mode=_flex_attention_compile_mode)
import torch.distributed as dist

import fastvideo.envs as envs
from fastvideo.attention import (DistributedAttention,
                                 LocalAttention)
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import (PatchEmbed)
from fastvideo.logger import init_logger
from fastvideo.models.dits._relative_rope import relativistic_window_offsets
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanT2VCrossAttention, WanTimeTextImageEmbedding
from fastvideo.platforms import AttentionBackendEnum, current_platform

logger = init_logger(__name__)

GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES = 21


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm=True,
                 eps=1e-6,
                 parallel_attention=False,
                 rope_cache_policy: str = "absolute") -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.rope_cache_policy = rope_cache_policy

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted((max(0, s), max(0, e)) for s, e in ranges):
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def _slice_ranges(x: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
        if len(ranges) == 1:
            start, end = ranges[0]
            return x[:, start:end].contiguous()
        return torch.cat([x[:, start:end] for start, end in ranges], dim=1).contiguous()

    def _context_ranges_for_block(
        self,
        block_end: int,
        context_limit: int,
        frame_seqlen: int,
    ) -> list[tuple[int, int]]:
        block_end = min(max(0, block_end), context_limit)
        if block_end <= 0:
            return []
        if self.local_attn_size == -1:
            return [(0, block_end)]

        ranges: list[tuple[int, int]] = []
        sink_end = min(max(0, self.sink_size) * frame_seqlen, block_end)
        if sink_end > 0:
            ranges.append((0, sink_end))

        window_tokens = max(0, self.local_attn_size - self.sink_size) * frame_seqlen
        if window_tokens > 0:
            ranges.append((max(0, block_end - window_tokens), block_end))
        return self._merge_ranges(ranges)

    def _blockwise_local_attention(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        frame_seqlen: int,
        num_frame_per_block: int,
        teacher_forcing_clean_len: int | None = None,
    ) -> torch.Tensor:
        block_tokens = frame_seqlen * num_frame_per_block
        if block_tokens <= 0:
            raise ValueError(f"block_tokens must be positive, got {block_tokens}")

        def attend_chunk(
            q_start: int,
            q_end: int,
            kv_ranges: list[tuple[int, int]],
        ) -> torch.Tensor:
            kv_ranges = self._merge_ranges(kv_ranges)
            if not kv_ranges:
                raise ValueError("local attention chunk has no KV context")
            return self.attn(
                roped_query[:, q_start:q_end].contiguous(),
                self._slice_ranges(roped_key, kv_ranges),
                self._slice_ranges(value, kv_ranges),
            )

        if teacher_forcing_clean_len is None:
            chunks = []
            seq_len = roped_query.shape[1]
            for block_start in range(0, seq_len, block_tokens):
                block_end = min(block_start + block_tokens, seq_len)
                chunks.append(attend_chunk(
                    block_start,
                    block_end,
                    self._context_ranges_for_block(block_end, seq_len, frame_seqlen),
                ))
            return torch.cat(chunks, dim=1)

        clean_len = int(teacher_forcing_clean_len)
        seq_len = roped_query.shape[1]
        if clean_len <= 0 or clean_len >= seq_len:
            raise ValueError(
                f"invalid teacher_forcing_clean_len={clean_len} for seq_len={seq_len}")

        clean_chunks = []
        noisy_chunks = []
        for block_start in range(0, clean_len, block_tokens):
            block_end = min(block_start + block_tokens, clean_len)
            clean_chunks.append(attend_chunk(
                block_start,
                block_end,
                self._context_ranges_for_block(block_end, clean_len, frame_seqlen),
            ))

        for block_start in range(0, clean_len, block_tokens):
            block_end = min(block_start + block_tokens, clean_len)
            clean_context_end = block_start
            kv_ranges = self._context_ranges_for_block(
                clean_context_end, clean_len, frame_seqlen)
            kv_ranges.append((clean_len + block_start, clean_len + block_end))
            noisy_chunks.append(attend_chunk(
                clean_len + block_start,
                clean_len + block_end,
                kv_ranges,
            ))

        return torch.cat(clean_chunks + noisy_chunks, dim=1)

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor],
                block_mask: BlockMask,
                kv_cache: dict | None = None,
                current_start: int = 0,
                cache_start: int | None = None,
                frame_seqlen: int = 1560,
                num_frame_per_block: int = 1,
                teacher_forcing_clean_len: int | None = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            frame_seqlen (int): Number of tokens per latent frame,
                e.g. 1560 for 480x832 resolution.
        """
        if cache_start is None:
            cache_start = current_start

        cos, sin = freqs_cis
        # relativistic defers roping until the cache window is known (and caches raw k)
        relativistic = self.rope_cache_policy == "relativistic" and kv_cache is not None
        if not relativistic:
            roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
            roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)

        if kv_cache is None and (
            self.local_attn_size != -1 or teacher_forcing_clean_len is not None
        ):
            x = self._blockwise_local_attention(
                roped_query,
                roped_key,
                v,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=num_frame_per_block,
                teacher_forcing_clean_len=teacher_forcing_clean_len,
            )
        elif kv_cache is None:
            # Padding for flex attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )
            if padded_length:
                x = x[:, :, :-padded_length]
            x = x.transpose(2, 1)
        else:
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            if self.local_attn_size == -1:
                max_attention_size = (GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES * frame_seqlen)
            else:
                max_attention_size = self.local_attn_size * frame_seqlen
            if self.local_attn_size == -1 and current_end > max_attention_size:
                raise ValueError(
                    "Causal Wan local_attn_size=-1 keeps the previous "
                    f"{GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES}-latent-frame KV "
                    "window for compatibility. Set local_attn_size for "
                    f"longer rollouts; got current_end={current_end} tokens "
                    f"with frame_seqlen={frame_seqlen}.")
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            stored_key = k if relativistic else roped_key  # raw vs roped in cache
            global_end_index = (
                int(kv_cache["global_end_index"].item())
                if isinstance(kv_cache["global_end_index"], torch.Tensor)
                else int(kv_cache["global_end_index"])
            )
            local_end_index_prev = (
                int(kv_cache["local_end_index"].item())
                if isinstance(kv_cache["local_end_index"], torch.Tensor)
                else int(kv_cache["local_end_index"])
            )
            if self.local_attn_size != -1 and (current_end > global_end_index) and (
                    num_new_tokens + local_end_index_prev > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + local_end_index_prev - kv_cache_size
                num_rolled_tokens = local_end_index_prev - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = local_end_index_prev + current_end - \
                    global_end_index - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = stored_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = local_end_index_prev + current_end - global_end_index
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"] = kv_cache["k"].detach()
                kv_cache["v"] = kv_cache["v"].detach()
                # logger.info("kv_cache['k'] is in comp graph: %s", kv_cache["k"].requires_grad or kv_cache["k"].grad_fn is not None)
                kv_cache["k"][:, local_start_index:local_end_index] = stored_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            key_window = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
            value_window = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
            if relativistic:
                window_len, query_lo, query_hi = relativistic_window_offsets(
                    local_end_index, num_new_tokens, max_attention_size)
                roped_query = _apply_rotary_emb(
                    q, cos[query_lo:query_hi], sin[query_lo:query_hi],
                    is_neox_style=False).type_as(v)
                key_window = _apply_rotary_emb(
                    key_window, cos[:window_len], sin[:window_len],
                    is_neox_style=False).type_as(v)
            x = self.attn(roped_query, key_window, value_window)
            if isinstance(kv_cache["global_end_index"], torch.Tensor):
                kv_cache["global_end_index"].fill_(current_end)
            else:
                kv_cache["global_end_index"] = current_end
            if isinstance(kv_cache["local_end_index"], torch.Tensor):
                kv_cache["local_end_index"].fill_(local_end_index)
            else:
                kv_cache["local_end_index"] = local_end_index

        return x

class CausalWanTransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = "",
                 rope_cache_policy: str = "absolute"):
        super().__init__()

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            rope_cache_policy=rope_cache_policy)
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32)

        # 2. Cross-attention
        # Only T2V for now
        self.attn2 = WanT2VCrossAttention(dim,
                                            num_heads,
                                            qk_norm=qk_norm,
                                            eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        frame_seqlen: int | None = None,
        num_frame_per_block: int = 1,
        teacher_forcing_clean_len: int | None = None,
    ) -> torch.Tensor:
        # hidden_states.shape: [batch_size, seq_length, inner_dim]
        # temb.shape: [batch_size, temb_seq_len, 6, inner_dim]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        temb_seq_len = temb.shape[1]
        tokens_per_temb = hidden_states.shape[1] // temb_seq_len
        if frame_seqlen is None:
            frame_seqlen = tokens_per_temb
        else:
            frame_seqlen = int(frame_seqlen)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb
        # e.shape: [batch_size, temb_seq_len, 6, inner_dim]
        assert e.shape == (bs, temb_seq_len, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2)
        # *_msa.shape: [batch_size, temb_seq_len, 1, inner_dim]
        # assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states).unflatten(dim=1, sizes=(temb_seq_len, tokens_per_temb)) *
                        (1 + scale_msa) + shift_msa).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            teacher_forcing_clean_len=teacher_forcing_clean_len,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        return hidden_states

class CausalWanTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig(
    )._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str,
                                                               Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size
        self.sink_size = config.sink_size
        self.rope_cache_policy = config.arch_config.rope_cache_policy

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            CausalWanTransformerBlock(inner_dim,
                              config.ffn_dim,
                              config.num_attention_heads,
                              config.local_attn_size,
                              config.sink_size,
                              config.qk_norm,
                              config.cross_attn_norm,
                              config.eps,
                              config.added_kv_proj_dim,
                              self._supported_attention_backends,
                              prefix=f"{config.prefix}.blocks.{i}",
                              rope_cache_policy=config.arch_config.rope_cache_policy)
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.block_mask = None
        self.teacher_forcing_block_mask = None
        self._logged_local_training_attention = False
        self._logged_relativistic_rope = False
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3
        self.independent_first_frame = False

        self.__post_init__()

    @staticmethod
    def _token_range_to_block_indices(start: int, end: int, block_size: int,
                                      max_blocks: int) -> list[int]:
        start = max(0, start)
        end = min(max(start, end), max_blocks * block_size)
        if end <= start:
            return []
        return list(range(start // block_size, (end - 1) // block_size + 1))

    @staticmethod
    def _block_mask_from_kv_rows(
        kv_rows: list[list[int]],
        q_len: int,
        kv_len: int,
        block_size: int,
        device: torch.device | str,
        mask_mod,
        all_blocks_full: bool,
    ) -> BlockMask:
        q_blocks = math.ceil(q_len / block_size)
        kv_blocks = math.ceil(kv_len / block_size)
        kv_num_blocks_cpu = torch.tensor(
            [len(row) for row in kv_rows], dtype=torch.int32
        ).view(1, 1, q_blocks)
        kv_indices_cpu = torch.zeros(
            (1, 1, q_blocks, kv_blocks), dtype=torch.int32
        )
        for row_idx, cols in enumerate(kv_rows):
            if cols:
                kv_indices_cpu[0, 0, row_idx, :len(cols)] = torch.tensor(
                    cols, dtype=torch.int32
                )

        kv_num_blocks = kv_num_blocks_cpu.to(device=device)
        kv_indices = kv_indices_cpu.to(device=device)
        full_kv_num_blocks = kv_num_blocks.clone() if all_blocks_full else None
        full_kv_indices = kv_indices.clone() if all_blocks_full else None
        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=mask_mod,
            seq_lengths=(q_len, kv_len),
        )

    @staticmethod
    def _context_ranges_for_block(
        block_end: int,
        context_limit: int,
        frame_seqlen: int,
        local_attn_size: int,
        sink_size: int,
    ) -> list[tuple[int, int]]:
        block_end = min(max(0, block_end), context_limit)
        if block_end <= 0:
            return []
        if local_attn_size == -1:
            return [(0, block_end)]

        ranges: list[tuple[int, int]] = []
        sink_end = min(max(0, sink_size) * frame_seqlen, block_end)
        if sink_end > 0:
            ranges.append((0, sink_end))

        window_tokens = max(0, local_attn_size - sink_size) * frame_seqlen
        if window_tokens > 0:
            ranges.append((max(0, block_end - window_tokens), block_end))
        return ranges

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1,
        sink_size=0
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                # Sliding window of size (local_attn_size - sink_size) frames plus
                # the first sink_size frames (attention sink), mirroring the rolling
                # KV cache at inference where sink + window == local_attn_size.
                sink_end = sink_size * frame_seqlen
                window_start = ends[q_idx] - (local_attn_size - sink_size) * frame_seqlen
                in_window = (kv_idx < ends[q_idx]) & (kv_idx >= window_start)
                in_sink = (kv_idx < ends[q_idx]) & (kv_idx < sink_end)
                return in_window | in_sink | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        q_len = total_length + padded_length
        kv_len = q_len
        block_size = 128
        q_blocks = math.ceil(q_len / block_size)
        kv_blocks = math.ceil(kv_len / block_size)
        all_blocks_full = (
            padded_length == 0
            and frame_seqlen % block_size == 0
            and (frame_seqlen * num_frame_per_block) % block_size == 0
        )
        kv_rows = []
        for q_block in range(q_blocks):
            q_start = q_block * block_size
            q_end = min(q_start + block_size, q_len)
            cols = []
            if q_start < total_length:
                q_last = min(q_end, total_length) - 1
                block_end = min(
                    ((q_last // (frame_seqlen * num_frame_per_block)) + 1)
                    * frame_seqlen * num_frame_per_block,
                    total_length,
                )
                if local_attn_size == -1:
                    cols.extend(CausalWanTransformer3DModel._token_range_to_block_indices(
                        0, block_end, block_size, kv_blocks))
                else:
                    window_tokens = max(0, local_attn_size - sink_size) * frame_seqlen
                    window_start = block_end - window_tokens
                    sink_end = min(sink_size * frame_seqlen, block_end)
                    cols.extend(CausalWanTransformer3DModel._token_range_to_block_indices(
                        window_start, block_end, block_size, kv_blocks))
                    cols.extend(CausalWanTransformer3DModel._token_range_to_block_indices(
                        0, sink_end, block_size, kv_blocks))
            cols.append(min(q_block, kv_blocks - 1))
            kv_rows.append(sorted(set(cols)))

        block_mask = CausalWanTransformer3DModel._block_mask_from_kv_rows(
            kv_rows=kv_rows,
            q_len=q_len,
            kv_len=kv_len,
            block_size=block_size,
            device=device,
            mask_mod=attention_mask,
            all_blocks_full=all_blocks_full,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(f"BlockMask(shape={block_mask.shape}, sparsity={block_mask.sparsity():.2f}%)")

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        local_attn_size=-1, sink_size=0
    ) -> BlockMask:
        """Attention mask for the teacher-forcing ``[clean | noisy]`` sequence.

        A noisy token attends to its own block plus the clean context of all
        strictly previous blocks; clean tokens are block-wise causal.
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        window_tokens = max(0, local_attn_size - sink_size) * frame_seqlen
        sink_tokens = max(0, sink_size) * frame_seqlen
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_sink_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_sink_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0, end=num_frames * frame_seqlen,
            step=attention_block_size, device=device, dtype=torch.long
        )
        for start_tensor in frame_indices:
            start = int(start_tensor.item())
            end = min(start + attention_block_size, clean_ends)
            context_ends[start:start + attention_block_size] = end
            if local_attn_size != -1:
                context_starts[start:start + attention_block_size] = max(0, end - window_tokens)
                context_sink_ends[start:start + attention_block_size] = min(sink_tokens, end)

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size, device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size
        for block_index, (start_tensor, end_tensor) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            clean_context_end = min(block_index * attention_block_size, clean_ends)
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = clean_context_end
            if local_attn_size != -1:
                noise_context_starts[start:end] = max(0, clean_context_end - window_tokens)
                noise_context_sink_ends[start:end] = min(sink_tokens, clean_context_end)

        def attention_mask(b, h, q_idx, kv_idx):
            clean_context = (
                ((kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx]))
                | (kv_idx < context_sink_ends[q_idx])
            )
            clean_mask = (q_idx < clean_ends) & clean_context
            c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            c2 = (
                ((kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx]))
                | (kv_idx < noise_context_sink_ends[q_idx])
            )
            noise_mask = (q_idx >= clean_ends) & (c1 | c2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        q_len = total_length + padded_length
        kv_len = q_len
        block_size = 128
        q_blocks = math.ceil(q_len / block_size)
        kv_blocks = math.ceil(kv_len / block_size)
        all_blocks_full = (
            padded_length == 0
            and frame_seqlen % block_size == 0
            and attention_block_size % block_size == 0
        )
        kv_rows = []
        for q_block in range(q_blocks):
            q_start = q_block * block_size
            q_end = min(q_start + block_size, q_len)
            cols = []

            if q_start < clean_ends:
                clean_last = min(q_end, clean_ends) - 1
                if clean_last >= q_start:
                    context_end = min(
                        ((clean_last // attention_block_size) + 1)
                        * attention_block_size,
                        clean_ends,
                    )
                    for range_start, range_end in CausalWanTransformer3DModel._context_ranges_for_block(
                        context_end, clean_ends, frame_seqlen,
                        local_attn_size, sink_size
                    ):
                        cols.extend(CausalWanTransformer3DModel._token_range_to_block_indices(
                            range_start, range_end, block_size, kv_blocks))

            if q_end > clean_ends and q_start < total_length:
                noisy_start = max(q_start, clean_ends)
                noisy_last = min(q_end, total_length) - 1
                if noisy_last >= noisy_start:
                    first_noise_block = (noisy_start - clean_ends) // attention_block_size
                    last_noise_block = (noisy_last - clean_ends) // attention_block_size
                    clean_context_end = min(
                        last_noise_block * attention_block_size, clean_ends
                    )
                    for range_start, range_end in CausalWanTransformer3DModel._context_ranges_for_block(
                        clean_context_end, clean_ends, frame_seqlen,
                        local_attn_size, sink_size
                    ):
                        cols.extend(CausalWanTransformer3DModel._token_range_to_block_indices(
                            range_start, range_end, block_size, kv_blocks))
                    for noise_block in range(first_noise_block, last_noise_block + 1):
                        noise_start = clean_ends + noise_block * attention_block_size
                        noise_end = min(noise_start + attention_block_size, total_length)
                        cols.extend(CausalWanTransformer3DModel._token_range_to_block_indices(
                            noise_start, noise_end, block_size, kv_blocks))

            cols.append(min(q_block, kv_blocks - 1))
            kv_rows.append(sorted(set(cols)))

        block_mask = CausalWanTransformer3DModel._block_mask_from_kv_rows(
            kv_rows=kv_rows,
            q_len=q_len,
            kv_len=kv_len,
            block_size=block_size,
            device=device,
            mask_mod=attention_mask,
            all_blocks_full=all_blocks_full,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a teacher-forcing mask with block size of {num_frame_per_block} frames")
            print(f"BlockMask(shape={block_mask.shape}, sparsity={block_mask.sparsity():.2f}%)")

        return block_mask

    def _forward_inference(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                kv_cache: dict = None,
                crossattn_cache: dict = None,
                current_start: int = 0,
                cache_start: int = 0,
                start_frame: int = 0,
                **kwargs) -> torch.Tensor:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)
        """

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        if self.rope_cache_policy == "relativistic":
            if not self._logged_relativistic_rope:
                logger.info(
                    "Using relativistic RoPE cache policy "
                    "(local_attn_size=%d, sink_size=%d)",
                    self.local_attn_size,
                    self.sink_size,
                )
                self._logged_relativistic_rope = True
            # fixed table over [0, max_attention_frames); attention slices it per step
            max_attention_frames = (
                GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES
                if self.local_attn_size == -1 else self.local_attn_size)
            rope_num_frames = max_attention_frames * get_sp_world_size()
            rope_start_frame = 0
        else:
            rope_num_frames = post_patch_num_frames * get_sp_world_size()
            rope_start_frame = start_frame  # 0 when kv_cache is None
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (rope_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=rope_start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos,
                     freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                        timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                    "frame_seqlen": post_patch_height * post_patch_width,
                }
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    **causal_kwargs)
            else:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                    "frame_seqlen": post_patch_height * post_patch_width,
                }
                hidden_states = block(hidden_states, encoder_hidden_states,
                                        timestep_proj, freqs_cis,
                                        **causal_kwargs)

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                    dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def _forward_train(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                start_frame: int = 0,
                clean_x: torch.Tensor | None = None,
                aug_t: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:

        orig_dtype = hidden_states.dtype
        teacher_forcing = clean_x is not None
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sp_world_size(), post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos,
                     freqs_sin) if freqs_cos is not None else None

        if teacher_forcing:
            if self.teacher_forcing_block_mask is None:
                self.teacher_forcing_block_mask = self._prepare_teacher_forcing_mask(
                    device=hidden_states.device,
                    num_frames=num_frames,
                    frame_seqlen=post_patch_height * post_patch_width,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size,
                )
            block_mask = self.teacher_forcing_block_mask
        else:
            if self.block_mask is None:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device=hidden_states.device,
                    num_frames=num_frames,
                    frame_seqlen=post_patch_height * post_patch_width,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size
                )
            block_mask = self.block_mask

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))], dim=1)
        encoder_hidden_states_text = encoder_hidden_states

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                        timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        if teacher_forcing:
            # Tile RoPE/modulation so clean frame i and noisy frame i share a position.
            clean_tokens = self.patch_embedding(clean_x).flatten(2).transpose(1, 2)
            teacher_forcing_clean_len = clean_tokens.shape[1]
            hidden_states = torch.cat([clean_tokens, hidden_states], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(timestep)
            _, timestep_proj_clean, _, _ = self.condition_embedder(
                aug_t.flatten(), encoder_hidden_states_text, None)
            timestep_proj_clean = timestep_proj_clean.unflatten(
                1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)
            timestep_proj = torch.cat([timestep_proj_clean, timestep_proj], dim=1)
            freqs_cis = (torch.cat([freqs_cos, freqs_cos], dim=0),
                         torch.cat([freqs_sin, freqs_sin], dim=0))

        # 4. Transformer blocks
        blockwise_training_attention = (
            self.local_attn_size != -1 or teacher_forcing_clean_len is not None
        )
        if blockwise_training_attention and not self._logged_local_training_attention:
            if not dist.is_initialized() or dist.get_rank() == 0:
                scope = (
                    "teacher-forcing full"
                    if self.local_attn_size == -1 and teacher_forcing_clean_len is not None
                    else "local"
                )
                print(
                    "Using FlashAttention blockwise training path "
                    f"({scope}, local_attn_size={self.local_attn_size}, sink_size={self.sink_size}, "
                    f"num_frame_per_block={self.num_frame_per_block})")
            self._logged_local_training_attention = True

        causal_kwargs = {
            "block_mask": block_mask,
            "frame_seqlen": post_patch_height * post_patch_width,
            "num_frame_per_block": self.num_frame_per_block,
            "teacher_forcing_clean_len": teacher_forcing_clean_len if teacher_forcing else None,
        }
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    **causal_kwargs)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                        timestep_proj, freqs_cis,
                                        **causal_kwargs)

        if teacher_forcing:
            hidden_states = hidden_states[:, hidden_states.shape[1] // 2:]

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                    dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)


    def unpatchify(self, x, grid_sizes):
        r"""


        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,


        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """

        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            # u = torch.einsum('fhwpqrc->cfphqwr', u.contiguous())
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

# Entry point for model registry
EntryClass = CausalWanTransformer3DModel
