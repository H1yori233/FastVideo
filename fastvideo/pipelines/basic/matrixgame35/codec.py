# SPDX-License-Identifier: Apache-2.0
"""Exact VAE boundaries used by Matrix-Game 3.5 rollout and Patch Memory."""

from __future__ import annotations

import numpy as np
import torch

MATRIXGAME35_VAE_SPATIAL_FACTOR = 16
MATRIXGAME35_VAE_ENCODE_TILE_SIZE = (34, 34)
MATRIXGAME35_VAE_ENCODE_TILE_STRIDE = (18, 16)
MATRIXGAME35_VAE_DECODE_TILE_SIZE = (30, 52)
MATRIXGAME35_VAE_DECODE_TILE_STRIDE = (15, 26)


def _matrixgame35_spatial_tile_tasks(
    height: int,
    width: int,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Build the released spatial tile list, including its tail-skip rule."""
    size_h, size_w = tile_size
    stride_h, stride_w = tile_stride
    if height <= 0 or width <= 0:
        raise ValueError("tiled VAE spatial dimensions must be positive.")
    if min(size_h, size_w, stride_h, stride_w) <= 0:
        raise ValueError("tiled VAE sizes and strides must be positive.")
    if stride_h > size_h or stride_w > size_w:
        raise ValueError("tiled VAE strides must not exceed tile sizes.")

    tasks = []
    for h in range(0, height, stride_h):
        if h - stride_h >= 0 and h - stride_h + size_h >= height:
            continue
        for w in range(0, width, stride_w):
            if w - stride_w >= 0 and w - stride_w + size_w >= width:
                continue
            tasks.append((h, h + size_h, w, w + size_w))
    return tuple(tasks)


def _matrixgame35_axis_mask(
    length: int,
    *,
    lower_bound: bool,
    upper_bound: bool,
    border_width: int,
) -> torch.Tensor:
    mask = torch.ones(length, dtype=torch.float32, device="cpu")
    if border_width < 0:
        raise ValueError(f"invalid tile border width {border_width} for length {length}.")
    if border_width == 0 or (lower_bound and upper_bound):
        return mask
    if border_width > length:
        raise ValueError(f"invalid tile border width {border_width} for length {length}.")
    ramp = (torch.arange(border_width, dtype=torch.float32) + 1) / border_width
    if not lower_bound:
        mask[:border_width] = ramp
    if not upper_bound:
        mask[-border_width:] = torch.flip(ramp, dims=(0, ))
    return mask


def _matrixgame35_tile_mask(
    tile: torch.Tensor,
    *,
    bounds: tuple[bool, bool, bool, bool],
    border_width: tuple[int, int],
) -> torch.Tensor:
    height, width = tile.shape[-2:]
    mask_h = _matrixgame35_axis_mask(
        height,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        border_width=border_width[0],
    ).view(height, 1)
    mask_w = _matrixgame35_axis_mask(
        width,
        lower_bound=bounds[2],
        upper_bound=bounds[3],
        border_width=border_width[1],
    ).view(1, width)
    return torch.minimum(mask_h, mask_w).view(1, 1, 1, height, width).to(dtype=tile.dtype)


def _matrixgame35_accumulate_tile(
    values: torch.Tensor,
    weight: torch.Tensor,
    tile: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    bounds: tuple[bool, bool, bool, bool],
    border_width: tuple[int, int],
) -> None:
    tile = tile.detach().to(device="cpu", dtype=values.dtype)
    if tile.ndim != 5 or tile.shape[0] != 1:
        raise ValueError(f"tiled VAE operation must return [1,C,T,H,W], got {tuple(tile.shape)}.")
    target_h_end = target_h + tile.shape[-2]
    target_w_end = target_w + tile.shape[-1]
    if target_h_end > values.shape[-2] or target_w_end > values.shape[-1]:
        raise ValueError("tiled VAE output exceeds the allocated spatial canvas.")
    mask = _matrixgame35_tile_mask(tile, bounds=bounds, border_width=border_width)
    values[:, :, :, target_h:target_h_end, target_w:target_w_end] += tile * mask
    weight[:, :, :, target_h:target_h_end, target_w:target_w_end] += mask


def _matrixgame35_finish_weighted_tiles(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if torch.any(weight == 0):
        raise RuntimeError("tiled VAE merge left uncovered output pixels.")
    return values / weight


def _matrixgame35_validate_spatial_factor(vae) -> None:
    config = getattr(vae, "config", None)
    factor = getattr(config, "scale_factor_spatial", None)
    if factor is None:
        factor = getattr(config, "spatial_compression_ratio", None)
    if factor is not None and int(factor) != MATRIXGAME35_VAE_SPATIAL_FACTOR:
        raise ValueError("Matrix-Game 3.5 tiled VAE requires spatial compression factor "
                         f"{MATRIXGAME35_VAE_SPATIAL_FACTOR}, got {factor}.")


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


def encode_matrixgame35_tiled_video(
    vae,
    video: torch.Tensor,
    *,
    tile_size: tuple[int, int] = MATRIXGAME35_VAE_ENCODE_TILE_SIZE,
    tile_stride: tuple[int, int] = MATRIXGAME35_VAE_ENCODE_TILE_STRIDE,
) -> torch.Tensor:
    """Encode one video with the released weighted spatial tiling contract."""
    if video.ndim != 5 or video.shape[:2] != (1, 3):
        raise ValueError(f"tiled video must have shape [1,3,T,H,W], got {tuple(video.shape)}.")
    _matrixgame35_validate_spatial_factor(vae)
    factor = MATRIXGAME35_VAE_SPATIAL_FACTOR
    _, _, num_frames, height, width = video.shape
    if height % factor or width % factor:
        raise ValueError(f"tiled video height and width must be divisible by {factor}.")

    sample_tile_size = (tile_size[0] * factor, tile_size[1] * factor)
    sample_tile_stride = (tile_stride[0] * factor, tile_stride[1] * factor)
    tasks = _matrixgame35_spatial_tile_tasks(
        height,
        width,
        sample_tile_size,
        sample_tile_stride,
    )
    z_dim = int(getattr(vae, "z_dim", 0))
    if z_dim <= 0:
        raise ValueError("Matrix-Game 3.5 VAE must expose a positive z_dim.")
    latent_frames = (num_frames + 3) // 4
    values = torch.zeros(
        (1, z_dim, latent_frames, height // factor, width // factor),
        dtype=video.dtype,
        device="cpu",
    )
    weight = torch.zeros(
        (1, 1, latent_frames, height // factor, width // factor),
        dtype=video.dtype,
        device="cpu",
    )
    border_width = (tile_size[0] - tile_stride[0], tile_size[1] - tile_stride[1])
    for h, h_end, w, w_end in tasks:
        tile = video[:, :, :, h:h_end, w:w_end]
        encoded = encode_matrixgame35_video(vae, tile)
        expected_shape = (
            1,
            z_dim,
            latent_frames,
            tile.shape[-2] // factor,
            tile.shape[-1] // factor,
        )
        if tuple(encoded.shape) != expected_shape:
            raise ValueError(f"tiled VAE encode returned {tuple(encoded.shape)}, expected {expected_shape}.")
        _matrixgame35_accumulate_tile(
            values,
            weight,
            encoded,
            target_h=h // factor,
            target_w=w // factor,
            bounds=(h == 0, h_end >= height, w == 0, w_end >= width),
            border_width=border_width,
        )
    return _matrixgame35_finish_weighted_tiles(values, weight)


def encode_matrixgame35_independent_frames(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode every ``[N,3,H,W]`` frame as its own one-frame video.

    The result is ``[N,C,1,H_lat,W_lat]``. Keeping frames in the batch
    dimension avoids accidental temporal VAE compression in Patch Memory.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must have shape [N,3,H,W], got {tuple(frames.shape)}.")
    if frames.shape[0] == 0:
        raise ValueError("frames must contain at least one image.")
    # The released wrapper passes a list of one-frame videos and its VAE
    # processes that list sequentially.  Keep the same execution boundary:
    # batching all 84 rollout frames through the native VAE at 704p has a much
    # larger peak-memory footprint even though the resulting tensor shape is
    # identical.
    latents = torch.cat(
        [encode_matrixgame35_video(vae,
                                   frame.unsqueeze(0).unsqueeze(2)) for frame in frames],
        dim=0,
    )
    if latents.shape[0] != frames.shape[0] or latents.shape[2] != 1:
        raise ValueError("independent frame encoding must produce one latent per input frame, got "
                         f"{tuple(latents.shape)} for {tuple(frames.shape)}.")
    return latents


def encode_matrixgame35_tiled_independent_frames(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode frames independently with the released spatial tiling contract."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must have shape [N,3,H,W], got {tuple(frames.shape)}.")
    if frames.shape[0] == 0:
        raise ValueError("frames must contain at least one image.")
    return torch.cat(
        [encode_matrixgame35_tiled_video(vae,
                                         frame.unsqueeze(0).unsqueeze(2)) for frame in frames],
        dim=0,
    )


def matrixgame35_memory_latents(vae, frames: torch.Tensor) -> torch.Tensor:
    """Return independently encoded memory frames as CPU FP32 ``[C,N,H,W]``."""
    latents = encode_matrixgame35_independent_frames(vae, frames)
    return latents[:, :, 0].permute(1, 0, 2, 3).detach().to(device="cpu", dtype=torch.float32).contiguous()


def matrixgame35_tiled_memory_latents(vae, frames: torch.Tensor) -> torch.Tensor:
    """Return independently tiled memory latents as CPU FP32 ``[C,N,H,W]``."""
    latents = encode_matrixgame35_tiled_independent_frames(vae, frames)
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


def decode_matrixgame35_tiled_video(
    vae,
    latents: torch.Tensor,
    *,
    tile_size: tuple[int, int] = MATRIXGAME35_VAE_DECODE_TILE_SIZE,
    tile_stride: tuple[int, int] = MATRIXGAME35_VAE_DECODE_TILE_STRIDE,
) -> torch.Tensor:
    """Decode one latent video with weighted spatial merge, then clamp once."""
    if latents.ndim != 5 or latents.shape[0] != 1:
        raise ValueError(f"tiled latents must have shape [1,C,T,H,W], got {tuple(latents.shape)}.")
    decode_unclamped = getattr(vae, "decode_unclamped", None)
    if not callable(decode_unclamped):
        raise TypeError("Matrix-Game 3.5 tiled decode requires vae.decode_unclamped().")
    _matrixgame35_validate_spatial_factor(vae)
    factor = MATRIXGAME35_VAE_SPATIAL_FACTOR
    _, _, latent_frames, height, width = latents.shape
    output_frames = (latent_frames - 1) * 4 + 1
    tasks = _matrixgame35_spatial_tile_tasks(height, width, tile_size, tile_stride)
    values = torch.zeros(
        (1, 3, output_frames, height * factor, width * factor),
        dtype=latents.dtype,
        device="cpu",
    )
    weight = torch.zeros(
        (1, 1, output_frames, height * factor, width * factor),
        dtype=latents.dtype,
        device="cpu",
    )
    border_width = (
        (tile_size[0] - tile_stride[0]) * factor,
        (tile_size[1] - tile_stride[1]) * factor,
    )
    for h, h_end, w, w_end in tasks:
        tile = denormalize_matrixgame35_latents(vae, latents[:, :, :, h:h_end, w:w_end])
        decoded = decode_unclamped(tile)
        expected_shape = (
            1,
            3,
            output_frames,
            tile.shape[-2] * factor,
            tile.shape[-1] * factor,
        )
        if not torch.is_tensor(decoded) or tuple(decoded.shape) != expected_shape:
            shape = tuple(decoded.shape) if torch.is_tensor(decoded) else type(decoded).__name__
            raise ValueError(f"tiled VAE decode returned {shape}, expected {expected_shape}.")
        _matrixgame35_accumulate_tile(
            values,
            weight,
            decoded,
            target_h=h * factor,
            target_w=w * factor,
            bounds=(h == 0, h_end >= height, w == 0, w_end >= width),
            border_width=border_width,
        )
    decoded = _matrixgame35_finish_weighted_tiles(values, weight).clamp_(-1.0, 1.0)
    return decoded.float().mul_(0.5).add_(0.5)


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
