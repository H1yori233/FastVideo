# SPDX-License-Identifier: Apache-2.0
"""Bidirectional Matrix-Game 3.5 transformer.

All released Matrix-Game 3.5 checkpoints share this Wan2.2-TI2V backbone.
Checkpoint-specific subject-reference tables are optional root parameters;
the causal execution path is intentionally a later milestone.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from fastvideo.configs.models.dits.matrixgame35 import MatrixGame35WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.rotary_embedding import _apply_rotary_emb, get_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.models.dits._matrixgame35_prope import prope_dot_product_attention
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanT2VCrossAttention, WanTimeTextImageEmbedding
from fastvideo.platforms import current_platform


_CAUSAL_FORWARD_KWARGS = frozenset({
    "block_mask",
    "cache_start",
    "causal_kv_config",
    "crossattn_cache",
    "current_start",
    "kv_cache",
    "start_frame",
})


def _sequence_parallel_world_size() -> int:
    """Treat an uninitialized distributed runtime as the single-rank case."""
    try:
        return get_sp_world_size()
    except AssertionError:
        return 1


def _extract_viewmats(camera_info: Sequence[Any] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if camera_info is None:
        raise ValueError("camera_info=(w2c, (P, P_T, P_inv)) is required.")
    if not isinstance(camera_info, Sequence) or len(camera_info) != 2:
        raise ValueError(
            "camera_info must contain exactly (w2c, (P, P_T, P_inv)); "
            "view-change and causal camera metadata are not supported yet."
        )
    viewmats = camera_info[1]
    if not isinstance(viewmats, Sequence) or len(viewmats) != 3:
        raise ValueError("camera_info[1] must contain exactly (P, P_T, P_inv).")
    if not all(isinstance(matrix, torch.Tensor) for matrix in viewmats):
        raise ValueError("P, P_T, and P_inv must be tensors.")
    return viewmats[0], viewmats[1], viewmats[2]


class MatrixGame35TransformerBlock(nn.Module):
    """Wan block with native 3D RoPE followed by full-layout PRoPE."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        *,
        prope_camera_layout: str = "full",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")
        if dim // num_heads % 64 != 0:
            raise ValueError("Matrix-Game 3.5 PRoPE requires head_dim to be a multiple of 64.")
        if qk_norm != "rms_norm_across_heads":
            raise ValueError("Matrix-Game 3.5 requires qk_norm='rms_norm_across_heads'.")
        if not cross_attn_norm:
            raise ValueError("Matrix-Game 3.5 requires cross_attn_norm=True.")
        if prope_camera_layout != "full":
            raise ValueError("Matrix-Game 3.5 requires prope_camera_layout='full'.")

        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.prope_camera_layout = prope_camera_layout

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(
            dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.to_q"
        )
        self.to_k = ReplicatedLinear(
            dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.to_k"
        )
        self.to_v = ReplicatedLinear(
            dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.to_v"
        )
        self.to_out = ReplicatedLinear(
            dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.to_out"
        )
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        self.attn2 = WanT2VCrossAttention(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn2",
        )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        self.ffn = MLP(
            dim,
            ffn_dim,
            act_type="gelu_pytorch_tanh",
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.mlp_residual = ScaleResidual()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        viewmats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden_dim], "
                f"got {tuple(hidden_states.shape)}."
            )
        if temb.ndim != 4 or temb.shape[:2] != hidden_states.shape[:2] or temb.shape[2:] != (6, self.hidden_dim):
            raise ValueError(
                "temb must have shape [batch, sequence, 6, hidden_dim], "
                f"got {tuple(temb.shape)}."
            )

        orig_dtype = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        query = self.norm_q(query).unflatten(2, (self.num_attention_heads, -1))
        key = self.norm_k(key).unflatten(2, (self.num_attention_heads, -1))
        value = value.unflatten(2, (self.num_attention_heads, -1))

        cos, sin = freqs_cis
        query = _apply_rotary_emb(query, cos, sin, is_neox_style=False)
        key = _apply_rotary_emb(key, cos, sin, is_neox_style=False)
        attn_output = prope_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            viewmats=viewmats,
            camera_layout=self.prope_camera_layout,
        ).transpose(1, 2)
        attn_output, _ = self.to_out(attn_output.flatten(2))

        null_modulation = hidden_states.new_zeros(1)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states,
            attn_output,
            gate_msa,
            null_modulation,
            null_modulation,
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        attn_output = self.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            context_lens=None,
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states,
            attn_output,
            1,
            shift_mlp,
            scale_mlp,
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        ff_output = self.ffn(norm_hidden_states)
        return self.mlp_residual(hidden_states, ff_output, gate_mlp).to(orig_dtype)


_DEFAULT_MATRIXGAME35_CONFIG = MatrixGame35WanVideoConfig()


class MatrixGame35Transformer3DModel(BaseDiT):
    """Shared bidirectional DiT for the released Matrix-Game 3.5 variants."""

    _fsdp_shard_conditions = _DEFAULT_MATRIXGAME35_CONFIG._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_MATRIXGAME35_CONFIG._compile_conditions
    _supported_attention_backends = _DEFAULT_MATRIXGAME35_CONFIG._supported_attention_backends
    param_names_mapping = _DEFAULT_MATRIXGAME35_CONFIG.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_MATRIXGAME35_CONFIG.reverse_param_names_mapping
    lora_param_names_mapping = _DEFAULT_MATRIXGAME35_CONFIG.lora_param_names_mapping

    def __init__(
        self,
        config: MatrixGame35WanVideoConfig,
        hf_config: dict[str, Any],
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        if _sequence_parallel_world_size() != 1:
            raise NotImplementedError("Matrix-Game 3.5 PRoPE does not support sequence parallelism yet.")
        if not config.use_prope:
            raise ValueError("Released Matrix-Game 3.5 variants require use_prope=True.")
        if config.image_dim is not None or config.added_kv_proj_dim is not None:
            raise ValueError("Released Matrix-Game 3.5 variants use text-only cross-attention.")

        self.quant_config = config.quant_config
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.causal = bool(config.causal)

        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
        )
        self.blocks = nn.ModuleList([
            MatrixGame35TransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                prope_camera_layout=config.prope_camera_layout,
                quant_config=config.quant_config,
                prefix=f"{config.prefix}.blocks.{index}",
            )
            for index in range(config.num_layers)
        ])
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        max_refs = int(config.subject_ref_memory_max_refs)
        self.subject_ref_memory_enabled = max_refs > 0
        if self.subject_ref_memory_enabled:
            self.subject_ref_memory_max_refs = max_refs
            self.subject_ref_memory_local_pos_size = 64
            self.subject_ref_index_embedding = nn.Parameter(
                self.scale_shift_table.new_zeros(max_refs, inner_dim)
            )
            self.subject_ref_type_embedding = nn.Parameter(
                self.scale_shift_table.new_zeros(1, inner_dim)
            )
            self.subject_ref_local_h_embedding = nn.Parameter(
                self.scale_shift_table.new_zeros(64, inner_dim)
            )
            self.subject_ref_local_w_embedding = nn.Parameter(
                self.scale_shift_table.new_zeros(64, inner_dim)
            )

        self.gradient_checkpointing = False
        self.__post_init__()

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        viewmats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[int, int, int, int]:
        if _sequence_parallel_world_size() != 1:
            raise NotImplementedError("Matrix-Game 3.5 PRoPE does not support sequence parallelism yet.")
        if hidden_states.ndim != 5:
            raise ValueError(
                "hidden_states must have shape [batch, channels, frames, height, width], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch_size, channels, frames, height, width = hidden_states.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {channels}.")
        p_t, p_h, p_w = self.patch_size
        if frames % p_t or height % p_h or width % p_w:
            raise ValueError(
                f"Latent shape {(frames, height, width)} must be divisible by patch_size {self.patch_size}."
            )
        grid = (frames // p_t, height // p_h, width // p_w)
        sequence_length = math.prod(grid)
        if timestep.ndim != 2 or timestep.shape != (batch_size, sequence_length):
            raise ValueError(
                "timestep must provide one value per packed latent token with shape "
                f"{(batch_size, sequence_length)}, got {tuple(timestep.shape)}."
            )
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError(
                "encoder_hidden_states must have shape [batch, text_sequence, text_dim] "
                f"with batch {batch_size}, got {tuple(encoder_hidden_states.shape)}."
            )

        expected_matrix_shape = (batch_size, grid[0], 4, 4, 4)
        for name, matrix in zip(("P", "P_T", "P_inv"), viewmats):
            if matrix.shape != expected_matrix_shape:
                raise ValueError(f"{name} must have shape {expected_matrix_shape}, got {tuple(matrix.shape)}.")
            if matrix.device != hidden_states.device or matrix.dtype != hidden_states.dtype:
                raise ValueError(f"{name} must match hidden_states device and dtype.")
        return batch_size, grid[0], grid[1], grid[2]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        *,
        camera_info: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del guidance
        if self.causal:
            raise NotImplementedError("Matrix-Game 3.5 causal forward is a later milestone.")
        causal_kwargs = sorted(name for name in kwargs if name in _CAUSAL_FORWARD_KWARGS)
        if causal_kwargs:
            raise NotImplementedError(
                "Matrix-Game 3.5 causal forward arguments are not supported yet: "
                + ", ".join(causal_kwargs)
            )
        if kwargs:
            raise TypeError(f"Unexpected Matrix-Game 3.5 forward arguments: {', '.join(sorted(kwargs))}.")
        if encoder_hidden_states_image is not None:
            raise ValueError("Released Matrix-Game 3.5 variants do not accept image encoder states.")
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) != 1:
                raise ValueError("encoder_hidden_states lists must contain exactly one tensor.")
            encoder_hidden_states = encoder_hidden_states[0]
        if not isinstance(encoder_hidden_states, torch.Tensor):
            raise ValueError("encoder_hidden_states must be a tensor.")

        viewmats = _extract_viewmats(camera_info)
        batch_size, grid_frames, grid_height, grid_width = self._validate_forward_inputs(
            hidden_states,
            encoder_hidden_states,
            timestep,
            viewmats,
        )
        orig_dtype = hidden_states.dtype
        sequence_length = grid_frames * grid_height * grid_width

        head_dim = self.hidden_size // self.num_attention_heads
        rope_dim_list = [
            head_dim - 4 * (head_dim // 6),
            2 * (head_dim // 6),
            2 * (head_dim // 6),
        ]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (grid_frames, grid_height, grid_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            rope_theta=10000,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
        )
        freqs_cis = (
            freqs_cos.to(device=hidden_states.device, dtype=torch.float32),
            freqs_sin.to(device=hidden_states.device, dtype=torch.float32),
        )

        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        temb = self.condition_embedder.time_embedder(timestep.flatten())
        temb = temb.unflatten(0, (batch_size, sequence_length))
        timestep_proj = self.condition_embedder.time_modulation(temb)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        assert self.condition_embedder.text_embedder is not None
        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)
        if encoder_hidden_states.dtype != orig_dtype:
            raise ValueError(
                "Embedded text and latent dtypes must match, got "
                f"{encoder_hidden_states.dtype} and {orig_dtype}."
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                viewmats,
            )

        shift, scale = (
            self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift.squeeze(2), scale.squeeze(2))
        hidden_states = self.proj_out(hidden_states)
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            grid_frames,
            grid_height,
            grid_width,
            p_t,
            p_h,
            p_w,
            self.out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = MatrixGame35Transformer3DModel
