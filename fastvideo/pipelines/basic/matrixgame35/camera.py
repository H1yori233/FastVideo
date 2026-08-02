# SPDX-License-Identifier: Apache-2.0
"""Camera input and PRoPE matrix preparation for Matrix-Game 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fastvideo.memory.matrixgame35.constants import (
    RGB_FRAMES_PER_BLOCK,
    RGB_SUBFRAMES_PER_LATENT,
)


@dataclass(frozen=True)
class MatrixGame35CameraTrajectory:
    """Canonical FP32 camera-to-world poses and pixel-space intrinsics."""

    c2w: torch.Tensor
    intrinsics: torch.Tensor

    def __post_init__(self) -> None:
        if self.c2w.ndim != 3 or self.c2w.shape[-2:] != (4, 4):
            raise ValueError(f"c2w must have shape [frames, 4, 4], got {tuple(self.c2w.shape)}.")
        if self.intrinsics.shape != (self.c2w.shape[0], 3, 3):
            raise ValueError("intrinsics must have shape [frames, 3, 3] matching c2w, "
                             f"got {tuple(self.intrinsics.shape)}.")
        if self.c2w.shape[0] == 0:
            raise ValueError("camera trajectory must contain at least one frame.")
        if self.c2w.dtype != torch.float32 or self.intrinsics.dtype != torch.float32:
            raise ValueError("camera trajectory tensors must remain FP32 before model preparation.")


def required_camera_frames(num_blocks: int) -> int:
    """Return the executable upstream contract: anchor plus 84 frames/block."""
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
    return 1 + RGB_FRAMES_PER_BLOCK * int(num_blocks)


def _intrinsics_to_matrices(intrinsics: np.ndarray, frame_count: int) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (4, ):
        intrinsics = intrinsics[None]
    if intrinsics.ndim == 2 and intrinsics.shape[-1] == 4:
        matrices = np.zeros((intrinsics.shape[0], 3, 3), dtype=np.float32)
        matrices[:, 0, 0] = intrinsics[:, 0]
        matrices[:, 1, 1] = intrinsics[:, 1]
        matrices[:, 0, 2] = intrinsics[:, 2]
        matrices[:, 1, 2] = intrinsics[:, 3]
        matrices[:, 2, 2] = 1.0
        intrinsics = matrices
    elif intrinsics.shape == (3, 3):
        intrinsics = intrinsics[None]
    elif intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape (4,), (N,4), (3,3), or (N,3,3), "
                         f"got {intrinsics.shape}.")
    if intrinsics.shape[0] == 0:
        raise ValueError("intrinsics must contain at least one frame.")
    if intrinsics.shape[0] < frame_count:
        intrinsics = np.concatenate(
            [intrinsics, np.repeat(intrinsics[-1:], frame_count - intrinsics.shape[0], axis=0)], axis=0)
    return intrinsics[:frame_count]


def _pad_last(array: np.ndarray, frame_count: int) -> np.ndarray:
    if array.shape[0] >= frame_count:
        return array[:frame_count]
    return np.concatenate([array, np.repeat(array[-1:], frame_count - array.shape[0], axis=0)], axis=0)


def load_camera_trajectory(
    path: str | Path,
    *,
    convention: str = "c2w",
    frame_count: int | None = None,
) -> MatrixGame35CameraTrajectory:
    """Load the public camera ``.npz`` contract and normalize it to C2W."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "extrinsics_c2w" in payload:
            extrinsics = np.asarray(payload["extrinsics_c2w"], dtype=np.float32)
        elif "extrinsics" in payload:
            extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
        else:
            raise ValueError(f"{path} must contain 'extrinsics_c2w' or 'extrinsics'.")
        if "intrinsics" not in payload:
            raise ValueError(f"{path} must contain 'intrinsics'.")
        intrinsics_raw = np.asarray(payload["intrinsics"], dtype=np.float32)

    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics must have shape (N,4,4), got {extrinsics.shape}.")
    if extrinsics.shape[0] == 0:
        raise ValueError("extrinsics must contain at least one frame.")
    convention = convention.strip().lower()
    if convention == "w2c":
        extrinsics = np.linalg.inv(extrinsics.astype(np.float64)).astype(np.float32)
    elif convention != "c2w":
        raise ValueError(f"camera convention must be 'c2w' or 'w2c', got {convention!r}.")

    intrinsics = _intrinsics_to_matrices(intrinsics_raw, extrinsics.shape[0])
    if frame_count is not None:
        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}.")
        extrinsics = _pad_last(extrinsics, int(frame_count))
        intrinsics = _pad_last(intrinsics, int(frame_count))
    return MatrixGame35CameraTrajectory(
        c2w=torch.from_numpy(np.ascontiguousarray(extrinsics)),
        intrinsics=torch.from_numpy(np.ascontiguousarray(intrinsics)),
    )


def normalize_matrixgame35_intrinsics(
    intrinsics: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    mode: str = "per_frame",
) -> np.ndarray:
    """Recenter pixel-space intrinsics with the released mosaic policy."""
    intrinsics = np.asarray(intrinsics)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must have shape [frames, 3, 3], got {intrinsics.shape}.")
    if intrinsics.shape[0] == 0:
        raise ValueError("intrinsics must contain at least one frame.")
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image dimensions must be positive, got {image_height}x{image_width}.")
    if mode not in ("per_frame", "first_frame", "episode_mean"):
        raise ValueError("intrinsics mode must be 'per_frame', 'first_frame', or 'episode_mean', "
                         f"got {mode!r}.")

    processed = intrinsics.astype(np.float32, copy=True)
    if mode == "first_frame":
        processed = np.repeat(processed[:1], repeats=processed.shape[0], axis=0)

    cx = processed[:, 0, 2]
    cy = processed[:, 1, 2]
    target_cx = float(image_width) * 0.5
    target_cy = float(image_height) * 0.5
    valid = (np.abs(cx) > 1e-6) & (np.abs(cy) > 1e-6)
    if np.any(valid):
        processed[valid, 0, 0] = processed[valid, 0, 0] / cx[valid] * target_cx
        processed[valid, 1, 1] = processed[valid, 1, 1] / cy[valid] * target_cy
        processed[valid, 0, 1] = processed[valid, 0, 1] / cx[valid] * target_cx
        processed[valid, 0, 2] = target_cx
        processed[valid, 1, 2] = target_cy

    if mode == "episode_mean":
        processed = np.repeat(processed.mean(axis=0, keepdims=True), repeats=processed.shape[0], axis=0)
    return processed


def gather_latent_subframes(
    trajectory: MatrixGame35CameraTrajectory,
    *,
    first_rgb_frame: int,
    latent_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather four consecutive RGB cameras for each requested latent frame."""
    if first_rgb_frame < 0:
        raise ValueError(f"first_rgb_frame must be non-negative, got {first_rgb_frame}.")
    if latent_count <= 0:
        raise ValueError(f"latent_count must be positive, got {latent_count}.")
    stop = first_rgb_frame + RGB_SUBFRAMES_PER_LATENT * int(latent_count)
    if stop > trajectory.c2w.shape[0]:
        raise ValueError(
            f"camera trajectory has {trajectory.c2w.shape[0]} frames but [{first_rgb_frame}:{stop}] was requested.")
    c2w = trajectory.c2w[first_rgb_frame:stop].reshape(latent_count, RGB_SUBFRAMES_PER_LATENT, 4, 4)
    intrinsics = trajectory.intrinsics[first_rgb_frame:stop].reshape(latent_count, RGB_SUBFRAMES_PER_LATENT, 3, 3)
    return c2w.unsqueeze(0), intrinsics.unsqueeze(0)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inverse = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rotation_inverse
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inverse, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(intrinsics.shape[:-2] + (4, 4), device=intrinsics.device, dtype=intrinsics.dtype)
    out[..., :3, :3] = intrinsics
    out[..., 3, 3] = 1.0
    return out


def _invert_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(intrinsics)
    out[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    out[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    out[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    out[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _scale_translation(translation: torch.Tensor, mode: float | str) -> torch.Tensor:
    if isinstance(mode, str):
        if mode != "logd4":
            raise ValueError(f"translation_scale must be a number or 'logd4', got {mode!r}.")
        norm = translation.norm(dim=-1, keepdim=True)
        return translation * (torch.log1p(norm) / norm.clamp_min(1e-8) / 4.0)
    scale = float(mode)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(f"numeric translation_scale must be finite and non-zero, got {mode!r}.")
    return translation / scale


def build_prope_viewmats(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    translation_scale: float | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the released full-layout PRoPE matrices in model dtype."""
    if c2w.ndim != 5 or c2w.shape[-3:] != (RGB_SUBFRAMES_PER_LATENT, 4, 4):
        raise ValueError(f"c2w must have shape [B,T,4,4,4], got {tuple(c2w.shape)}.")
    if intrinsics.shape != c2w.shape[:-2] + (3, 3):
        raise ValueError("intrinsics must have shape [B,T,4,3,3] matching c2w, "
                         f"got {tuple(intrinsics.shape)}.")
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image dimensions must be positive, got {image_height}x{image_width}.")
    if c2w.dtype != torch.float32 or intrinsics.dtype != torch.float32:
        raise ValueError("absolute c2w and intrinsics inputs must be FP32.")

    # Match upstream: invert the canonical C2W carrier before the model-dtype
    # camera unit, then perform normalization and projection composition in the
    # requested DiT dtype.
    inverse_input = c2w.cpu() if c2w.device.type == "mps" else c2w
    w2c = torch.linalg.inv(inverse_input.double()).float().to(device=c2w.device, dtype=dtype)
    intrinsics_model = intrinsics.to(device=w2c.device, dtype=dtype)
    w2c[..., :3, 3] = _scale_translation(w2c[..., :3, 3], translation_scale)

    normalized = torch.zeros_like(intrinsics_model)
    normalized[..., 0, 0] = intrinsics_model[..., 0, 0] / float(image_width)
    normalized[..., 1, 1] = intrinsics_model[..., 1, 1] / float(image_height)
    normalized[..., 0, 2] = intrinsics_model[..., 0, 2] / float(image_width) - 0.5
    normalized[..., 1, 2] = intrinsics_model[..., 1, 2] / float(image_height) - 0.5
    normalized[..., 2, 2] = 1.0

    projection = torch.einsum("...ij,...jk->...ik", _lift_intrinsics(normalized), w2c)
    projection_transpose = projection.transpose(-1, -2)
    projection_inverse = torch.einsum("...ij,...jk->...ik", _invert_se3(w2c),
                                      _lift_intrinsics(_invert_intrinsics(normalized)))
    return projection, projection_transpose, projection_inverse
