# SPDX-License-Identifier: Apache-2.0
"""Small policy helpers for Matrix-Game 3.5 distilled runtime profiles."""

from __future__ import annotations

import math

import torch

from fastvideo.configs.pipelines.matrixgame35 import matrixgame35_distilled_profile_settings


def distilled_profile_guidance_scale(profile: str) -> float:
    """Return the executable release guidance contract for one profile."""
    matrixgame35_distilled_profile_settings(profile)
    # The pinned upstream rollout rejects HiAR when guidance is not 1. Other
    # released profiles use guidance 3.
    return 1.0 if profile == "hiar-sde" else 3.0


def distilled_hiar_noise_seed(
    base_seed: int,
    *,
    batch_index: int,
    chunk_index: int,
    step_index: int,
    dynamic_context: bool,
) -> int:
    """Derive the pinned rollout's independent HiAR prefix-noise stream."""
    offset = 8_000_000 if dynamic_context else 7_000_000
    return (int(base_seed) + int(batch_index) * 1_000_000 + int(chunk_index) * 10_000 + int(step_index) * 101 + offset)


def make_distilled_hiar_noise(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Sample with the same device-local generator used by the upstream wrapper."""
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )


def hiar_sde_corrupt_clean_latents(
    clean_latents: torch.Tensor,
    context_sigma: torch.Tensor | float,
    *,
    keep_first_clean: bool,
    corruption_scale: float = 1.0,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Re-noise detached clean prefix latents at the next schedule level."""
    scale = float(corruption_scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(f"HiAR corruption_scale must be finite and lie in [0, 1], got {scale}.")
    clean = clean_latents.detach()
    sigma = torch.as_tensor(context_sigma, device=clean.device)
    if float(sigma.detach().float()) <= 0.0 or scale <= 0.0:
        return clean.clone()
    noise = noise.to(device=clean.device, dtype=clean.dtype)
    if noise.shape != clean.shape:
        raise ValueError(f"HiAR noise shape must match clean latents: {tuple(noise.shape)} vs {tuple(clean.shape)}.")
    scheduled = (1.0 - sigma) * clean + sigma * noise
    corrupted = scheduled if scale >= 1.0 else clean + scale * (scheduled - clean)
    if keep_first_clean and corrupted.shape[2] > 0:
        corrupted[:, :, :1] = clean[:, :, :1]
    return corrupted


def trim_distilled_rolling_latents(
    latents: torch.Tensor,
    *,
    frames_per_chunk: int,
    window_chunks: int,
) -> torch.Tensor:
    """Mirror moving-anchor cache eviction for HiAR latent provenance."""
    if latents.ndim != 5:
        raise ValueError(f"rolling latents must be [B,C,T,H,W], got {tuple(latents.shape)}.")
    frames_per_chunk = int(frames_per_chunk)
    window_chunks = int(window_chunks)
    if frames_per_chunk <= 0 or window_chunks <= 0:
        raise ValueError("frames_per_chunk and window_chunks must be positive.")
    max_frames = 1 + (window_chunks - 1) * frames_per_chunk
    output = latents
    while int(output.shape[2]) > max_frames:
        total = int(output.shape[2])
        if total <= frames_per_chunk:
            return output[:, :, -max_frames:].contiguous()
        keep = torch.as_tensor(
            [frames_per_chunk, *range(1 + frames_per_chunk, total)],
            device=output.device,
            dtype=torch.long,
        )
        output = output.index_select(2, keep).contiguous()
    return output


__all__ = [
    "distilled_hiar_noise_seed",
    "distilled_profile_guidance_scale",
    "hiar_sde_corrupt_clean_latents",
    "make_distilled_hiar_noise",
    "trim_distilled_rolling_latents",
]
