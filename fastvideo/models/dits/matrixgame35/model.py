# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 transformer.

All released Matrix-Game 3.5 checkpoints share this Wan2.2-TI2V backbone.
Checkpoint-specific subject-reference tables are optional root parameters;
the distilled variant uses the same parameters with causal K/V-cache execution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

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
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.models.dits.matrixgame35.causal_attention import (
    MatrixGame35CausalKVCache,
    init_matrixgame35_causal_kv_caches,
    matrixgame35_causal_kv_attention,
)
from fastvideo.models.dits.matrixgame35.conditioning import (
    build_mosaic_cross_attention_keep_mask,
    build_subject_ref_memory_tokens,
    prepend_subject_ref_prope_camera_info,
)
from fastvideo.models.dits.matrixgame35.prope import prope_dot_product_attention
from fastvideo.models.dits.matrixgame35.rope import (
    apply_matrixgame35_rope,
    build_matrixgame35_rope_frequencies,
    matrixgame35_rope_tables,
)
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanT2VCrossAttention, WanTimeTextImageEmbedding


class MatrixGame35LatentLayoutProtocol(Protocol):
    """Structural model input owned and materialized by the pipeline layer."""

    latents: torch.Tensor
    first_frame_count: int
    mosaic_frame_count: int
    noisy_frame_count: int
    latent_rope_time_indices: torch.Tensor
    token_timesteps: torch.Tensor
    mosaic_hole_mask: torch.Tensor | None
    drop_mosaic_holes: bool
    cross_attention_keep_mask: torch.Tensor | None
    subject_ref_prefix_token_count: int


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


def _select_drop_hole_viewmats(
    viewmats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    keep_indices: torch.Tensor,
    latent_keep_indices: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    full_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the per-token PRoPE carriers retained by hole dropping."""

    selected = []
    for matrix in viewmats:
        camera_length = int(matrix.shape[1])
        if camera_length == full_sequence_length:
            selected.append(matrix.index_select(1, keep_indices).contiguous())
            continue
        if camera_length != frame_count:
            raise ValueError(
                "PRoPE viewmats must be full-sequence token carriers or one carrier per latent frame, got "
                f"{camera_length} for sequence {full_sequence_length} and {frame_count} frames."
            )
        token_viewmats = (
            matrix.unsqueeze(2)
            .expand(matrix.shape[0], frame_count, tokens_per_frame, *matrix.shape[2:])
            .reshape(matrix.shape[0], frame_count * tokens_per_frame, *matrix.shape[2:])
        )
        selected.append(token_viewmats.index_select(1, latent_keep_indices).contiguous())
    return selected[0], selected[1], selected[2]


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
        rope_frequencies: torch.Tensor,
        viewmats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_keep_mask: torch.Tensor | None = None,
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

        query = apply_matrixgame35_rope(query, rope_frequencies)
        key = apply_matrixgame35_rope(key, rope_frequencies)
        attn_output = prope_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            viewmats=viewmats,
            camera_layout=self.prope_camera_layout,
            attn_mask=self_attention_mask,
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
        if cross_attention_keep_mask is not None:
            if cross_attention_keep_mask.ndim != 1 or cross_attention_keep_mask.shape[0] != hidden_states.shape[1]:
                raise ValueError(
                    "cross_attention_keep_mask must have one value per token, got "
                    f"{tuple(cross_attention_keep_mask.shape)} for sequence {hidden_states.shape[1]}."
                )
            attn_output = attn_output * cross_attention_keep_mask.to(
                device=attn_output.device,
                dtype=attn_output.dtype,
            ).view(1, -1, 1)
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

    def forward_causal(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rope_frequencies: torch.Tensor,
        cache_rope_frequencies: torch.Tensor | None,
        viewmats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        mosaic_token_count: int,
        current_positions: Sequence[int] | torch.Tensor,
        current_frames: Sequence[int] | torch.Tensor,
        mosaic_frames: Sequence[int] | torch.Tensor,
        cache: MatrixGame35CausalKVCache,
        cache_frames: Sequence[int] | torch.Tensor,
        cache_read_chunk_id: int | None,
        current_cache_chunk_ids: Sequence[int] | torch.Tensor | None,
        write_cache: bool,
        mosaic_hole_keep: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one block while keeping the optional mosaic prefix frozen."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim].")
        if temb.ndim != 4 or temb.shape[:2] != hidden_states.shape[:2] or temb.shape[2:] != (6, self.hidden_dim):
            raise ValueError("temb must have shape [batch, sequence, 6, hidden_dim].")

        orig_dtype = hidden_states.dtype
        frozen_mosaic = hidden_states[:, :mosaic_token_count].clone() if mosaic_token_count else None
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
        attn_output = matrixgame35_causal_kv_attention(
            query,
            key,
            value,
            rope_frequencies=rope_frequencies,
            cache_rope_frequencies=cache_rope_frequencies,
            viewmats=viewmats,
            camera_layout=self.prope_camera_layout,
            mosaic_token_count=mosaic_token_count,
            current_positions=current_positions,
            current_frames=current_frames,
            mosaic_frames=mosaic_frames,
            cache=cache,
            cache_frames=cache_frames,
            cache_read_chunk_id=cache_read_chunk_id,
            current_cache_chunk_ids=current_cache_chunk_ids,
            write_cache=write_cache,
            mosaic_hole_keep=mosaic_hole_keep,
        )
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
        hidden_states = self.mlp_residual(hidden_states, ff_output, gate_mlp).to(orig_dtype)
        if frozen_mosaic is not None:
            hidden_states = torch.cat((frozen_mosaic, hidden_states[:, mosaic_token_count:]), dim=1)
        return hidden_states


_DEFAULT_MATRIXGAME35_CONFIG = MatrixGame35WanVideoConfig()


class MatrixGame35Transformer3DModel(BaseDiT):
    """Shared DiT for the released bidirectional and distilled causal variants."""

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

        valid_camera_lengths = (grid[0], sequence_length)
        for name, matrix in zip(("P", "P_T", "P_inv"), viewmats):
            if (
                matrix.ndim != 5
                or matrix.shape[0] != batch_size
                or matrix.shape[1] not in valid_camera_lengths
                or matrix.shape[2:] != (4, 4, 4)
            ):
                raise ValueError(
                    f"{name} must be per-frame or per-token with shape "
                    f"[{batch_size}, {grid[0]} or {sequence_length}, 4, 4, 4], "
                    f"got {tuple(matrix.shape)}."
                )
            if matrix.device != hidden_states.device or matrix.dtype != hidden_states.dtype:
                raise ValueError(f"{name} must match hidden_states device and dtype.")
        return batch_size, grid[0], grid[1], grid[2]

    def init_causal_kv_caches(self) -> list[MatrixGame35CausalKVCache]:
        """Create empty per-block caches for the distilled execution path."""
        return init_matrixgame35_causal_kv_caches(len(self.blocks))

    @staticmethod
    def _causal_int_list(
        values: Sequence[int] | torch.Tensor | None,
        name: str,
    ) -> list[int] | None:
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.detach().reshape(-1).cpu().tolist()
        try:
            return [int(value) for value in values]
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional sequence of integers.") from exc

    def _forward_causal(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        camera_info: Sequence[Any] | None,
        kv_caches: list[MatrixGame35CausalKVCache] | None,
        mosaic_latents: torch.Tensor | None,
        current_positions: Sequence[int] | torch.Tensor | None,
        current_frames: Sequence[int] | torch.Tensor | None,
        mosaic_positions: Sequence[int] | torch.Tensor | None,
        mosaic_frames: Sequence[int] | torch.Tensor | None,
        cache_positions: Sequence[int] | torch.Tensor | None,
        cache_frames: Sequence[int] | torch.Tensor | None,
        cache_read_chunk_id: int | None,
        current_cache_chunk_ids: Sequence[int] | torch.Tensor | None,
        write_cache: bool,
    ) -> torch.Tensor:
        """Run one distilled autoregressive chunk and return only current frames."""
        if kv_caches is None or len(kv_caches) != len(self.blocks):
            raise ValueError(
                f"kv_caches must contain one cache per transformer block ({len(self.blocks)})."
            )
        if hidden_states.ndim != 5:
            raise ValueError("hidden_states must have shape [B, C, T, H, W].")
        input_batch, channels, current_frame_count, latent_height, latent_width = hidden_states.shape
        if channels != self.in_channels or current_frame_count <= 0:
            raise ValueError(
                f"Expected non-empty {self.in_channels}-channel current latents, got {tuple(hidden_states.shape)}."
            )
        if mosaic_latents is not None:
            expected = (input_batch, channels, latent_height, latent_width)
            actual = (
                mosaic_latents.shape[0],
                mosaic_latents.shape[1],
                mosaic_latents.shape[3],
                mosaic_latents.shape[4],
            ) if mosaic_latents.ndim == 5 else None
            if actual != expected:
                raise ValueError(
                    "mosaic_latents must share current batch/channel/spatial shape, got "
                    f"{tuple(mosaic_latents.shape)} vs {tuple(hidden_states.shape)}."
                )
            mosaic_latents = mosaic_latents.to(device=hidden_states.device, dtype=hidden_states.dtype)
            combined_latents = torch.cat((mosaic_latents, hidden_states), dim=2)
            mosaic_frame_count = int(mosaic_latents.shape[2])
        else:
            combined_latents = hidden_states
            mosaic_frame_count = 0

        current_positions_list = self._causal_int_list(current_positions, "current_positions")
        current_frames_list = self._causal_int_list(current_frames, "current_frames")
        if current_positions_list is None or current_frames_list is None:
            raise ValueError("current_positions and current_frames are required for causal execution.")
        if len(current_positions_list) != current_frame_count or len(current_frames_list) != current_frame_count:
            raise ValueError(
                "current_positions/current_frames length must match the current latent frame count, got "
                f"{len(current_positions_list)}/{len(current_frames_list)} vs {current_frame_count}."
            )

        mosaic_positions_list = self._causal_int_list(mosaic_positions, "mosaic_positions")
        mosaic_frames_list = self._causal_int_list(mosaic_frames, "mosaic_frames")
        if mosaic_frame_count:
            if mosaic_positions_list is None:
                mosaic_positions_list = list(mosaic_frames_list or [])
            if mosaic_frames_list is None:
                mosaic_frames_list = list(mosaic_positions_list)
            if len(mosaic_positions_list) != mosaic_frame_count or len(mosaic_frames_list) != mosaic_frame_count:
                raise ValueError(
                    "mosaic_positions/mosaic_frames length must match mosaic_latents, got "
                    f"{len(mosaic_positions_list)}/{len(mosaic_frames_list)} vs {mosaic_frame_count}."
                )
        else:
            if mosaic_positions_list or mosaic_frames_list:
                raise ValueError("mosaic positions/frames require mosaic_latents.")
            mosaic_positions_list = []
            mosaic_frames_list = []

        cache_positions_list = self._causal_int_list(cache_positions, "cache_positions")
        cache_frames_list = self._causal_int_list(cache_frames, "cache_frames")
        first_cache = kv_caches[0]
        if cache_positions_list is None:
            cache_positions_list = list(first_cache.get("positions", []))
        if cache_frames_list is None:
            cache_frames_list = list(first_cache.get("frames", []))
        if not cache_positions_list and cache_frames_list:
            cache_positions_list = list(cache_frames_list)
        if not cache_frames_list and cache_positions_list:
            cache_frames_list = list(cache_positions_list)
        if len(cache_positions_list) != len(cache_frames_list):
            raise ValueError("cache_positions and cache_frames must have the same length.")

        p_t, p_h, p_w = self.patch_size
        total_frames = mosaic_frame_count + current_frame_count
        if total_frames % p_t or latent_height % p_h or latent_width % p_w:
            raise ValueError(
                f"Latent shape {(total_frames, latent_height, latent_width)} must be divisible by {self.patch_size}."
            )
        grid_frames = total_frames // p_t
        grid_height = latent_height // p_h
        grid_width = latent_width // p_w
        tokens_per_frame = grid_height * grid_width

        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) != 1:
                raise ValueError("encoder_hidden_states lists must contain exactly one tensor.")
            encoder_hidden_states = encoder_hidden_states[0]
        if encoder_hidden_states.ndim != 3:
            raise ValueError("encoder_hidden_states must have shape [B, text_sequence, text_dim].")
        batch_size = int(encoder_hidden_states.shape[0])
        if input_batch not in (1, batch_size):
            raise ValueError("Current latent batch must be 1 or match encoder_hidden_states batch.")

        viewmats = _extract_viewmats(camera_info)
        all_camera_frames = cache_frames_list + mosaic_frames_list + current_frames_list
        if not all_camera_frames or min(all_camera_frames) < 0:
            raise ValueError("Causal PRoPE frame indices must be non-negative.")
        for name, matrix in zip(("P", "P_T", "P_inv"), viewmats):
            if (
                matrix.ndim != 5
                or matrix.shape[0] not in (1, batch_size)
                or matrix.shape[2:] != (4, 4, 4)
                or max(all_camera_frames) >= matrix.shape[1]
            ):
                raise ValueError(
                    f"{name} must cover every explicit causal camera frame for batch {batch_size}, "
                    f"got {tuple(matrix.shape)} and max frame {max(all_camera_frames)}."
                )
            if matrix.device != hidden_states.device or matrix.dtype != hidden_states.dtype:
                raise ValueError(f"{name} must match hidden_states device and dtype.")

        frame_timestep = torch.as_tensor(timestep, device=hidden_states.device, dtype=hidden_states.dtype)
        if frame_timestep.ndim == 0:
            frame_timestep = frame_timestep.reshape(1)
        if frame_timestep.ndim == 1:
            if frame_timestep.numel() != total_frames:
                raise ValueError(f"Causal timestep must have one value per input frame ({total_frames}).")
            frame_timestep = frame_timestep.reshape(1, total_frames)
        elif frame_timestep.ndim == 2:
            if frame_timestep.shape[1] != total_frames or frame_timestep.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"Causal timestep must have shape [1 or {batch_size}, {total_frames}], "
                    f"got {tuple(frame_timestep.shape)}."
                )
        else:
            raise ValueError("Causal timestep must be a scalar, vector, or rank-2 frame tensor.")
        frame_timestep = frame_timestep.expand(batch_size, -1)
        token_timestep = frame_timestep.repeat_interleave(tokens_per_frame, dim=1)

        hidden_states = self.patch_embedding(combined_latents).flatten(2).transpose(1, 2)
        if input_batch == 1 and batch_size > 1:
            hidden_states = hidden_states.repeat(batch_size, 1, 1)
        mosaic_token_count = mosaic_frame_count * tokens_per_frame
        mosaic_hole_keep = None
        mosaic_hole_mask = None
        if mosaic_latents is not None and mosaic_token_count:
            all_zero = (mosaic_latents == 0).all(dim=(0, 1))
            hole_patch = all_zero.reshape(
                mosaic_frame_count,
                latent_height // p_h,
                p_h,
                latent_width // p_w,
                p_w,
            ).all(dim=(2, 4)).flatten()
            mosaic_hole_keep = ~hole_patch
            mosaic_hole_mask = torch.zeros(
                total_frames * tokens_per_frame,
                device=hidden_states.device,
                dtype=torch.bool,
            )
            mosaic_hole_mask[:mosaic_token_count] = hole_patch.to(hidden_states.device)
            hidden_states = hidden_states.masked_fill(mosaic_hole_mask.view(1, -1, 1), 0)
            token_timestep = token_timestep.clone()
            token_timestep[:, mosaic_hole_mask] = 1000

        head_dim = self.hidden_size // self.num_attention_heads
        rope_tables = matrixgame35_rope_tables(head_dim)
        rope_frequencies = build_matrixgame35_rope_frequencies(
            rope_tables,
            torch.as_tensor(mosaic_positions_list + current_positions_list, dtype=torch.long),
            height=grid_height,
            width=grid_width,
            device=hidden_states.device,
        )
        if mosaic_hole_mask is not None:
            rope_frequencies = rope_frequencies.masked_fill(mosaic_hole_mask.view(-1, 1, 1), 0)
        cache_rope_frequencies = None
        if cache_positions_list:
            cache_rope_frequencies = build_matrixgame35_rope_frequencies(
                rope_tables,
                torch.as_tensor(cache_positions_list, dtype=torch.long),
                height=grid_height,
                width=grid_width,
                device=hidden_states.device,
            )

        temb = self.condition_embedder.time_embedder(token_timestep.flatten())
        temb = temb.unflatten(0, (batch_size, total_frames * tokens_per_frame))
        timestep_proj = self.condition_embedder.time_modulation(temb).unflatten(2, (6, -1))
        assert self.condition_embedder.text_embedder is not None
        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)
        if encoder_hidden_states.dtype != hidden_states.dtype:
            raise ValueError("Embedded text and latent dtypes must match.")

        for block, cache in zip(self.blocks, kv_caches):
            hidden_states = block.forward_causal(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rope_frequencies,
                cache_rope_frequencies,
                viewmats,
                mosaic_token_count=mosaic_token_count,
                current_positions=current_positions_list,
                current_frames=current_frames_list,
                mosaic_frames=mosaic_frames_list,
                cache=cache,
                cache_frames=cache_frames_list,
                cache_read_chunk_id=cache_read_chunk_id,
                current_cache_chunk_ids=current_cache_chunk_ids,
                write_cache=write_cache,
                mosaic_hole_keep=mosaic_hole_keep,
            )

        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift.squeeze(2), scale.squeeze(2))
        hidden_states = self.proj_out(hidden_states)
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
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states[:, :, mosaic_frame_count:]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        *,
        camera_info: Sequence[Any] | None = None,
        latent_layout: MatrixGame35LatentLayoutProtocol | None = None,
        subject_ref_latents: torch.Tensor | None = None,
        subject_ref_slot_ratio: float = 0.5,
        subject_ref_time_gap: int = 1,
        subject_ref_prope_mode: str = "identity",
        height: int | None = None,
        width: int | None = None,
        kv_caches: list[MatrixGame35CausalKVCache] | None = None,
        mosaic_latents: torch.Tensor | None = None,
        current_positions: Sequence[int] | torch.Tensor | None = None,
        current_frames: Sequence[int] | torch.Tensor | None = None,
        mosaic_positions: Sequence[int] | torch.Tensor | None = None,
        mosaic_frames: Sequence[int] | torch.Tensor | None = None,
        cache_positions: Sequence[int] | torch.Tensor | None = None,
        cache_frames: Sequence[int] | torch.Tensor | None = None,
        cache_read_chunk_id: int | None = None,
        current_cache_chunk_ids: Sequence[int] | torch.Tensor | None = None,
        write_cache: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the full noncausal layout or only the current causal chunk."""
        del guidance
        if self.causal:
            if latent_layout is not None or subject_ref_latents is not None:
                raise ValueError("The distilled causal checkpoint does not use noncausal layout or subject references.")
            if kwargs:
                raise TypeError(f"Unexpected Matrix-Game 3.5 forward arguments: {', '.join(sorted(kwargs))}.")
            if encoder_hidden_states_image is not None:
                raise ValueError("Released Matrix-Game 3.5 variants do not accept image encoder states.")
            return self._forward_causal(
                hidden_states,
                encoder_hidden_states,
                timestep,
                camera_info=camera_info,
                kv_caches=kv_caches,
                mosaic_latents=mosaic_latents,
                current_positions=current_positions,
                current_frames=current_frames,
                mosaic_positions=mosaic_positions,
                mosaic_frames=mosaic_frames,
                cache_positions=cache_positions,
                cache_frames=cache_frames,
                cache_read_chunk_id=cache_read_chunk_id,
                current_cache_chunk_ids=current_cache_chunk_ids,
                write_cache=write_cache,
            )
        causal_arguments_present = any(
            value is not None
            for value in (
                kv_caches,
                mosaic_latents,
                current_positions,
                current_frames,
                mosaic_positions,
                mosaic_frames,
                cache_positions,
                cache_frames,
                cache_read_chunk_id,
                current_cache_chunk_ids,
            )
        ) or write_cache
        if causal_arguments_present:
            raise NotImplementedError("Causal K/V-cache arguments require a causal Matrix-Game 3.5 config.")
        causal_kwargs = sorted(name for name in kwargs if name in _CAUSAL_FORWARD_KWARGS)
        if causal_kwargs:
            raise NotImplementedError(
                "Use the explicit Matrix-Game 3.5 causal cache arguments instead of: "
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

        if latent_layout is not None:
            if hidden_states.shape != latent_layout.latents.shape:
                raise ValueError(
                    "hidden_states and latent_layout.latents must have the same shape, got "
                    f"{tuple(hidden_states.shape)} and {tuple(latent_layout.latents.shape)}."
                )
            if (
                hidden_states.device != latent_layout.latents.device
                or hidden_states.dtype != latent_layout.latents.dtype
            ):
                raise ValueError("hidden_states and latent_layout.latents must share device and dtype.")
            if int(hidden_states.shape[2]) != (
                latent_layout.first_frame_count
                + latent_layout.mosaic_frame_count
                + latent_layout.noisy_frame_count
            ):
                raise ValueError("latent_layout frame counts do not match hidden_states.")
            timestep = latent_layout.token_timesteps.reshape(1, -1).expand(hidden_states.shape[0], -1)

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
        rope_tables = matrixgame35_rope_tables(head_dim)
        if latent_layout is None:
            rope_time_indices = torch.arange(grid_frames, dtype=torch.long)
        else:
            rope_time_indices = latent_layout.latent_rope_time_indices
        rope_frequencies = build_matrixgame35_rope_frequencies(
            rope_tables,
            rope_time_indices,
            height=grid_height,
            width=grid_width,
            device=hidden_states.device,
        )

        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        subject_ref_memory = build_subject_ref_memory_tokens(
            self,
            subject_ref_latents,
            batch_size=batch_size,
            video_h=int(height) if height is not None else int(grid_height * self.patch_size[1] * 16),
            video_w=int(width) if width is not None else int(grid_width * self.patch_size[2] * 16),
            subject_ref_slot_ratio=subject_ref_slot_ratio,
            subject_ref_time_gap=subject_ref_time_gap,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            rope_frequencies=rope_tables,
        )
        subject_ref_token_count = 0 if subject_ref_memory is None else int(subject_ref_memory["token_count"])
        if latent_layout is not None and latent_layout.subject_ref_prefix_token_count not in (
            0,
            subject_ref_token_count,
        ):
            raise ValueError(
                "latent_layout subject_ref_prefix_token_count does not match the built subject memory, got "
                f"{latent_layout.subject_ref_prefix_token_count} and {subject_ref_token_count}."
            )
        if subject_ref_memory is not None:
            hidden_states = torch.cat((subject_ref_memory["x"], hidden_states), dim=1)
            rope_frequencies = torch.cat(
                (subject_ref_memory["freqs"].to(device=hidden_states.device), rope_frequencies),
                dim=0,
            )
            camera_info = prepend_subject_ref_prope_camera_info(
                tuple(camera_info) if camera_info is not None else None,
                prefix_token_count=subject_ref_token_count,
                tokens_per_frame=grid_height * grid_width,
                frame_count=grid_frames,
                mode=subject_ref_prope_mode,
                clean_anchor_token_index=(
                    max(0, latent_layout.first_frame_count - 1) * grid_height * grid_width
                    if latent_layout is not None
                    else 0
                ),
            )
            viewmats = _extract_viewmats(camera_info)

        full_timestep = timestep
        if subject_ref_token_count:
            ref_timestep = torch.zeros(
                batch_size,
                subject_ref_token_count,
                device=timestep.device,
                dtype=timestep.dtype,
            )
            full_timestep = torch.cat((ref_timestep, timestep), dim=1)

        cross_attention_keep_mask = None
        if latent_layout is not None and (
            latent_layout.mosaic_frame_count > 0 or subject_ref_token_count > 0
        ):
            if (
                latent_layout.cross_attention_keep_mask is not None
                and latent_layout.subject_ref_prefix_token_count == subject_ref_token_count
            ):
                cross_attention_keep_mask = latent_layout.cross_attention_keep_mask
            else:
                cross_attention_keep_mask = build_mosaic_cross_attention_keep_mask(
                    prefix_memory_token_count=subject_ref_token_count,
                    reference_token_count=0,
                    first_frame_count=latent_layout.first_frame_count,
                    mosaic_frame_count=latent_layout.mosaic_frame_count,
                    noisy_frame_count=latent_layout.noisy_frame_count,
                    tokens_per_frame=grid_height * grid_width,
                    device=hidden_states.device,
                )
        elif subject_ref_token_count:
            cross_attention_keep_mask = build_mosaic_cross_attention_keep_mask(
                prefix_memory_token_count=subject_ref_token_count,
                reference_token_count=0,
                first_frame_count=0,
                mosaic_frame_count=0,
                noisy_frame_count=grid_frames,
                tokens_per_frame=grid_height * grid_width,
                device=hidden_states.device,
            )

        self_attention_mask = None
        latent_keep_indices = None
        if latent_layout is not None and latent_layout.mosaic_hole_mask is not None:
            if latent_layout.mosaic_hole_mask.shape != (sequence_length,):
                raise ValueError(
                    "latent_layout.mosaic_hole_mask must have one value per latent token, got "
                    f"{tuple(latent_layout.mosaic_hole_mask.shape)} for {sequence_length}."
                )
            latent_hole_mask = latent_layout.mosaic_hole_mask.to(device=hidden_states.device, dtype=torch.bool)
            hole_mask = latent_hole_mask
            if subject_ref_token_count:
                hole_mask = torch.cat(
                    (
                        torch.zeros(subject_ref_token_count, device=hidden_states.device, dtype=torch.bool),
                        hole_mask,
                    )
                )
            if latent_layout.drop_mosaic_holes:
                keep_indices = torch.nonzero(~hole_mask, as_tuple=False).squeeze(-1)
                latent_keep_indices = torch.nonzero(~latent_hole_mask, as_tuple=False).squeeze(-1)
                full_sequence_length = sequence_length + subject_ref_token_count
                hidden_states = hidden_states.index_select(1, keep_indices)
                rope_frequencies = rope_frequencies.index_select(0, keep_indices)
                full_timestep = full_timestep.index_select(1, keep_indices)
                viewmats = _select_drop_hole_viewmats(
                    viewmats,
                    keep_indices=keep_indices,
                    latent_keep_indices=latent_keep_indices,
                    frame_count=grid_frames,
                    tokens_per_frame=grid_height * grid_width,
                    full_sequence_length=full_sequence_length,
                )
                if cross_attention_keep_mask is not None:
                    cross_attention_keep_mask = cross_attention_keep_mask.index_select(0, keep_indices)
            else:
                hidden_states = hidden_states.masked_fill(hole_mask.view(1, -1, 1), 0)
                rope_frequencies = rope_frequencies.masked_fill(hole_mask.view(-1, 1, 1), 0)
                self_attention_mask = (~hole_mask).view(1, 1, 1, -1)

        if cross_attention_keep_mask is not None:
            cross_attention_keep_mask = cross_attention_keep_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )
            if cross_attention_keep_mask.shape != (hidden_states.shape[1],):
                raise ValueError(
                    "cross-attention keep mask does not match the full sequence, got "
                    f"{tuple(cross_attention_keep_mask.shape)} for {hidden_states.shape[1]}."
                )

        full_sequence_length = hidden_states.shape[1]
        temb = self.condition_embedder.time_embedder(full_timestep.flatten())
        temb = temb.unflatten(0, (batch_size, full_sequence_length))
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
                rope_frequencies,
                viewmats,
                self_attention_mask,
                cross_attention_keep_mask,
            )

        if subject_ref_token_count:
            hidden_states = hidden_states[:, subject_ref_token_count:]
            temb = temb[:, subject_ref_token_count:]

        shift, scale = (
            self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift.squeeze(2), scale.squeeze(2))
        hidden_states = self.proj_out(hidden_states)
        if latent_keep_indices is not None:
            full_hidden_states = hidden_states.new_zeros(batch_size, sequence_length, hidden_states.shape[2])
            full_hidden_states.index_copy_(1, latent_keep_indices, hidden_states)
            hidden_states = full_hidden_states
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
