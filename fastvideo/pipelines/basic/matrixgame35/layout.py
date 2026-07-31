# SPDX-License-Identifier: Apache-2.0
"""Non-causal latent sequence layout for Matrix-Game 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from fastvideo.pipelines.basic.matrixgame35.conditioning import (
    build_mosaic_cross_attention_keep_mask, )


@dataclass(frozen=True)
class MatrixGame35LatentLayout:
    """Prepared ``clean + mosaic + noisy`` inputs and token metadata.

    Subject-reference tokens are built separately. ``subject_ref_prefix_token_count``
    is used only to align the transformer-level cross-attention mask.
    """

    latents: torch.Tensor
    first_frame_count: int
    mosaic_frame_count: int
    noisy_frame_count: int
    mosaic_frame_indices: torch.Tensor
    latent_rope_time_indices: torch.Tensor
    token_timesteps: torch.Tensor
    mosaic_hole_mask: torch.Tensor | None
    drop_mosaic_holes: bool
    cross_attention_keep_mask: torch.Tensor | None
    tokens_per_frame: int
    subject_ref_prefix_token_count: int

    @property
    def condition_frame_count(self) -> int:
        return self.first_frame_count + self.mosaic_frame_count

    @property
    def output_frame_slice(self) -> slice:
        """Noisy frames to retain after unpatchifying the model output."""
        return slice(self.condition_frame_count, self.latents.shape[2])

    @property
    def output_token_slice(self) -> slice:
        """Noisy tokens within the latent-token sequence (prefix excluded)."""
        start = self.condition_frame_count * self.tokens_per_frame
        return slice(start, start + self.noisy_frame_count * self.tokens_per_frame)


def _check_latents(name: str, value: torch.Tensor, noisy_latents: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W], got {tuple(value.shape)}.")
    if value.shape[:2] != noisy_latents.shape[:2] or value.shape[-2:] != noisy_latents.shape[-2:]:
        raise ValueError(f"{name} must share batch/channel/spatial shape with noisy_latents, got "
                         f"{tuple(value.shape)} vs {tuple(noisy_latents.shape)}.")
    return value.to(device=noisy_latents.device, dtype=noisy_latents.dtype)


def _resolve_mosaic_frame_indices(
    mosaic_frame_indices: torch.Tensor | Sequence[int] | None,
    *,
    noisy_frame_count: int,
    mosaic_frame_count: int,
    device: torch.device,
) -> torch.Tensor:
    if mosaic_frame_count <= 0:
        return torch.empty((0, ), dtype=torch.long, device=device)
    if mosaic_frame_indices is None:
        if mosaic_frame_count != noisy_frame_count:
            raise ValueError("mosaic_frame_indices is required when mosaic_frame_count "
                             f"({mosaic_frame_count}) differs from noisy_frame_count ({noisy_frame_count}).")
        return torch.arange(noisy_frame_count, dtype=torch.long, device=device)
    indices = torch.as_tensor(mosaic_frame_indices, device=device, dtype=torch.long).reshape(-1)
    if indices.numel() != mosaic_frame_count:
        raise ValueError("mosaic_frame_indices length must match mosaic_frame_count, got "
                         f"{indices.numel()} vs {mosaic_frame_count}.")
    if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= noisy_frame_count):
        raise ValueError(f"mosaic_frame_indices must be within the noisy latent range [0, {noisy_frame_count}).")
    return indices


def _resolve_latent_rope_time_indices(
    latent_rope_time_indices: torch.Tensor | Sequence[int] | None,
    *,
    first_frame_count: int,
    mosaic_frame_count: int,
    noisy_frame_count: int,
    mosaic_frame_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    total_frame_count = first_frame_count + mosaic_frame_count + noisy_frame_count
    if latent_rope_time_indices is not None:
        indices = torch.as_tensor(latent_rope_time_indices, device=device, dtype=torch.long).reshape(-1)
        if indices.numel() != total_frame_count:
            raise ValueError("latent_rope_time_indices length must match "
                             "first_frame_count + mosaic_frame_count + noisy_frame_count, got "
                             f"{indices.numel()} vs {total_frame_count}.")
        if indices.numel() and indices.min().item() < 0:
            raise ValueError("latent_rope_time_indices must be non-negative.")
        return indices

    first_times = torch.arange(first_frame_count, dtype=torch.long, device=device)
    mosaic_times = first_frame_count + mosaic_frame_indices
    noisy_times = first_frame_count + torch.arange(noisy_frame_count, dtype=torch.long, device=device)
    return torch.cat((first_times, mosaic_times, noisy_times))


def _build_mosaic_hole_mask(
    mosaic_latents: torch.Tensor | None,
    *,
    first_frame_count: int,
    noisy_frame_count: int,
    tokens_per_frame: int,
) -> torch.Tensor | None:
    if mosaic_latents is None:
        return None
    all_zero = (mosaic_latents == 0).all(dim=(0, 1))
    mosaic_frame_count, latent_height, latent_width = all_zero.shape
    hole_patches = (all_zero.reshape(mosaic_frame_count, latent_height // 2, 2, latent_width // 2,
                                     2).all(dim=(2, 4)).flatten())
    return torch.cat((
        torch.zeros(first_frame_count * tokens_per_frame, dtype=torch.bool, device=hole_patches.device),
        hole_patches,
        torch.zeros(noisy_frame_count * tokens_per_frame, dtype=torch.bool, device=hole_patches.device),
    ))


def build_noncausal_latent_layout(
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    first_frame_latents: torch.Tensor | None = None,
    mosaic_latents: torch.Tensor | None = None,
    mosaic_frame_indices: torch.Tensor | Sequence[int] | None = None,
    latent_rope_time_indices: torch.Tensor | Sequence[int] | None = None,
    subject_ref_prefix_token_count: int = 0,
    mask_mosaic_holes: bool = True,
    drop_mosaic_holes: bool = False,
    sequence_parallel_size: int = 1,
) -> MatrixGame35LatentLayout:
    """Prepare the released non-causal Matrix-Game 3.5 sequence contract.

    Latents are ordered as ``clean prefix + mosaic + noisy``. Clean and mosaic
    tokens receive timestep zero, noisy tokens receive ``timestep``. Matching
    upstream inference, all-zero mosaic patches are marked as holes and receive
    timestep 1000. Released Base inference physically drops those tokens inside
    the transformer when ``drop_mosaic_holes`` is enabled.
    """
    if sequence_parallel_size != 1:
        raise ValueError("Matrix-Game 3.5 latent layout currently supports sequence_parallel_size=1 only.")
    if noisy_latents.ndim != 5:
        raise ValueError(f"noisy_latents must have shape [B, C, T, H, W], got {tuple(noisy_latents.shape)}.")
    if noisy_latents.shape[2] <= 0:
        raise ValueError("noisy_latents must contain at least one latent frame.")
    latent_height, latent_width = noisy_latents.shape[-2:]
    if latent_height % 2 or latent_width % 2:
        raise ValueError("Matrix-Game 3.5 1x2x2 patch layout requires even latent height and width, got "
                         f"{latent_height}x{latent_width}.")
    if subject_ref_prefix_token_count < 0:
        raise ValueError("subject_ref_prefix_token_count must be non-negative.")

    sequence = []
    if first_frame_latents is not None:
        first_frame_latents = _check_latents("first_frame_latents", first_frame_latents, noisy_latents)
        sequence.append(first_frame_latents)
    first_frame_count = 0 if first_frame_latents is None else int(first_frame_latents.shape[2])

    if mosaic_latents is not None:
        mosaic_latents = _check_latents("mosaic_latents", mosaic_latents, noisy_latents)
        if mosaic_latents.shape[2] > noisy_latents.shape[2]:
            raise ValueError("mosaic_latents temporal length must be <= noisy_latents temporal length, got "
                             f"{mosaic_latents.shape[2]} vs {noisy_latents.shape[2]}.")
        sequence.append(mosaic_latents)
    mosaic_frame_count = 0 if mosaic_latents is None else int(mosaic_latents.shape[2])
    noisy_frame_count = int(noisy_latents.shape[2])
    resolved_mosaic_indices = _resolve_mosaic_frame_indices(
        mosaic_frame_indices,
        noisy_frame_count=noisy_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        device=noisy_latents.device,
    )

    sequence.append(noisy_latents)
    latents = noisy_latents if len(sequence) == 1 else torch.cat(sequence, dim=2)
    tokens_per_frame = latent_height * latent_width // 4
    condition_frame_count = first_frame_count + mosaic_frame_count
    timestep_scalar = torch.as_tensor(timestep, device=noisy_latents.device, dtype=noisy_latents.dtype).reshape(-1)[0]
    frame_timesteps = torch.cat((
        torch.zeros(condition_frame_count, dtype=noisy_latents.dtype, device=noisy_latents.device),
        timestep_scalar.expand(noisy_frame_count),
    ))
    token_timesteps = frame_timesteps.repeat_interleave(tokens_per_frame)

    mosaic_hole_mask = (_build_mosaic_hole_mask(
        mosaic_latents,
        first_frame_count=first_frame_count,
        noisy_frame_count=noisy_frame_count,
        tokens_per_frame=tokens_per_frame,
    ) if mask_mosaic_holes else None)
    if mosaic_hole_mask is not None:
        token_timesteps = token_timesteps.clone()
        token_timesteps[mosaic_hole_mask] = 1000

    rope_time_indices = _resolve_latent_rope_time_indices(
        latent_rope_time_indices,
        first_frame_count=first_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        noisy_frame_count=noisy_frame_count,
        mosaic_frame_indices=resolved_mosaic_indices,
        device=noisy_latents.device,
    )
    cross_attention_keep_mask = None
    if mosaic_frame_count > 0 or subject_ref_prefix_token_count > 0:
        cross_attention_keep_mask = build_mosaic_cross_attention_keep_mask(
            prefix_memory_token_count=subject_ref_prefix_token_count,
            reference_token_count=0,
            first_frame_count=first_frame_count,
            mosaic_frame_count=mosaic_frame_count,
            noisy_frame_count=noisy_frame_count,
            tokens_per_frame=tokens_per_frame,
            device=noisy_latents.device,
        )

    return MatrixGame35LatentLayout(
        latents=latents,
        first_frame_count=first_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        noisy_frame_count=noisy_frame_count,
        mosaic_frame_indices=resolved_mosaic_indices,
        latent_rope_time_indices=rope_time_indices,
        token_timesteps=token_timesteps,
        mosaic_hole_mask=mosaic_hole_mask,
        drop_mosaic_holes=bool(drop_mosaic_holes),
        cross_attention_keep_mask=cross_attention_keep_mask,
        tokens_per_frame=tokens_per_frame,
        subject_ref_prefix_token_count=subject_ref_prefix_token_count,
    )
