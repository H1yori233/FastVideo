# SPDX-License-Identifier: Apache-2.0
"""Exact VAE boundaries used by Matrix-Game 3.5 rollout and Patch Memory."""

from __future__ import annotations

import numpy as np
import torch


def _latent_stats(vae, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    config = getattr(vae, "config", None)
    means = getattr(config, "latents_mean", None)
    stds = getattr(config, "latents_std", None)
    if means is None or stds is None:
        raise ValueError("Matrix-Game 3.5 VAE config must define latents_mean and latents_std.")
    mean = torch.as_tensor(means, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(stds, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1, 1)
    if mean.shape[1] != tensor.shape[1] or std.shape[1] != tensor.shape[1]:
        raise ValueError(f"VAE latent statistics have {mean.shape[1]} channels but tensor has {tensor.shape[1]}.")
    return mean, std


def normalize_matrixgame35_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Apply the released ``(z - mean) / std`` latent normalization."""
    mean, std = _latent_stats(vae, latents)
    return (latents - mean) / std


def denormalize_matrixgame35_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Apply the released ``z * std + mean`` decoder boundary."""
    mean, std = _latent_stats(vae, latents)
    return latents * std + mean


def _posterior_mode(posterior) -> torch.Tensor:
    if hasattr(posterior, "mode"):
        return posterior.mode()
    if hasattr(posterior, "latent_dist") and hasattr(posterior.latent_dist, "mode"):
        return posterior.latent_dist.mode()
    if hasattr(posterior, "mean") and torch.is_tensor(posterior.mean):
        return posterior.mean
    raise TypeError("VAE encode output must expose mode(), latent_dist.mode(), or a tensor mean.")


def encode_matrixgame35_video(vae, video: torch.Tensor) -> torch.Tensor:
    """Deterministically encode normalized ``[B,3,T,H,W]`` RGB video."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"video must have shape [B,3,T,H,W], got {tuple(video.shape)}.")
    return normalize_matrixgame35_latents(vae, _posterior_mode(vae.encode(video)))


def encode_matrixgame35_independent_frames(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode every ``[N,3,H,W]`` frame as its own one-frame video.

    The result is ``[N,C,1,H_lat,W_lat]``. Keeping frames in the batch
    dimension avoids accidental temporal VAE compression in Patch Memory.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must have shape [N,3,H,W], got {tuple(frames.shape)}.")
    if frames.shape[0] == 0:
        raise ValueError("frames must contain at least one image.")
    latents = encode_matrixgame35_video(vae, frames.unsqueeze(2))
    if latents.shape[0] != frames.shape[0] or latents.shape[2] != 1:
        raise ValueError("independent frame encoding must produce one latent per input frame, got "
                         f"{tuple(latents.shape)} for {tuple(frames.shape)}.")
    return latents


def matrixgame35_memory_latents(vae, frames: torch.Tensor) -> torch.Tensor:
    """Return independently encoded memory frames as CPU FP32 ``[C,N,H,W]``."""
    latents = encode_matrixgame35_independent_frames(vae, frames)
    return latents[:, :, 0].permute(1, 0, 2, 3).detach().to(device="cpu", dtype=torch.float32).contiguous()


def decode_matrixgame35_video(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode normalized latents to clamped float RGB in ``[0,1]``."""
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape [B,C,T,H,W], got {tuple(latents.shape)}.")
    decoded = vae.decode(denormalize_matrixgame35_latents(vae, latents))
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    elif isinstance(decoded, tuple | list):
        decoded = decoded[0]
    if not torch.is_tensor(decoded) or decoded.ndim != 5:
        raise ValueError("VAE decode output must be a five-dimensional tensor.")
    return (decoded.float() * 0.5 + 0.5).clamp(0.0, 1.0)


def matrixgame35_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    """Convert ``[1,3,T,H,W]`` float RGB to contiguous ``[T,H,W,3]`` uint8."""
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"video must have shape [1,3,T,H,W], got {tuple(video.shape)}.")
    return (video[0].permute(1, 2, 3, 0).mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy())


def matrixgame35_uint8_to_frames(frames: np.ndarray, *, device: torch.device | str) -> torch.Tensor:
    """Convert contiguous ``[N,H,W,3]`` uint8 RGB to normalized ``[N,3,H,W]``."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"frames must have shape [N,H,W,3] and dtype uint8, got {array.shape} {array.dtype}.")
    return (torch.from_numpy(np.ascontiguousarray(array)).to(device=device,
                                                             dtype=torch.float32).permute(0, 3, 1,
                                                                                          2).mul(2.0 / 255.0).sub(1.0))
