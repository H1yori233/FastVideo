# SPDX-License-Identifier: Apache-2.0
"""Arbitrary-position native 3D RoPE for Matrix-Game 3.5."""

from functools import lru_cache

import torch


def _precompute_1d(
    dim: int,
    end: int,
    theta: float,
) -> torch.Tensor:
    indices = torch.arange(0, dim, 2, dtype=torch.float64, device="cpu")[:dim // 2]
    inverse_frequencies = 1.0 / (theta**(indices / dim))
    angles = torch.outer(
        torch.arange(end, dtype=torch.float64, device="cpu"),
        inverse_frequencies,
    )
    return torch.polar(torch.ones_like(angles), angles)


@lru_cache(maxsize=8)
def matrixgame35_rope_tables(
    head_dim: int,
    end: int = 1024,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the three CPU complex-frequency tables used by the released model."""
    head_dim = int(head_dim)
    end = int(end)
    if head_dim <= 0 or end <= 0:
        raise ValueError(f"head_dim and end must be positive, got {head_dim} and {end}.")

    temporal_dim = head_dim - 2 * (head_dim // 3)
    spatial_dim = head_dim // 3
    tables = (
        _precompute_1d(temporal_dim, end, theta),
        _precompute_1d(spatial_dim, end, theta),
        _precompute_1d(spatial_dim, end, theta),
    )
    complex_width = sum(int(table.shape[1]) for table in tables)
    if complex_width * 2 != head_dim:
        raise ValueError(
            "Matrix-Game 3.5 native RoPE requires its three complex bands to "
            f"cover head_dim exactly, got {complex_width * 2} for {head_dim}."
        )
    return tables


def build_matrixgame35_rope_frequencies(
    tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    time_indices: torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build ``[time * height * width, 1, head_dim / 2]`` complex RoPE."""
    if len(tables) != 3:
        raise ValueError("Matrix-Game 3.5 native RoPE requires three frequency tables.")
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height} and {width}.")

    temporal_table, height_table, width_table = tables
    table_device = temporal_table.device
    indices = torch.as_tensor(time_indices, device=table_device, dtype=torch.long).reshape(-1)
    if indices.numel() == 0:
        raise ValueError("time_indices must contain at least one frame.")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(temporal_table.shape[0]):
        raise ValueError(
            "time_indices must be within the temporal RoPE table, got range "
            f"[{int(indices.min().item())}, {int(indices.max().item())}] for "
            f"length {int(temporal_table.shape[0])}."
        )
    if height > int(height_table.shape[0]) or width > int(width_table.shape[0]):
        raise ValueError(
            "Spatial grid exceeds the native RoPE tables, got "
            f"{height}x{width} for {int(height_table.shape[0])}x{int(width_table.shape[0])}."
        )

    frames = int(indices.numel())
    frequencies = torch.cat(
        (
            temporal_table[indices].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            height_table[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            width_table[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ),
        dim=-1,
    )
    return frequencies.reshape(frames * height * width, 1, -1).to(device)


def apply_matrixgame35_rope(
    tensor: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Apply upstream-compatible complex RoPE to ``[B, S, H, D]`` tensors."""
    if tensor.ndim != 4:
        raise ValueError(f"tensor must have shape [B, S, H, D], got {tuple(tensor.shape)}.")
    if tensor.shape[-1] % 2:
        raise ValueError(f"RoPE head dimension must be even, got {tensor.shape[-1]}.")
    if frequencies.ndim != 3 or frequencies.shape[1] != 1:
        raise ValueError(
            "frequencies must have shape [sequence, 1, head_dim / 2], "
            f"got {tuple(frequencies.shape)}."
        )
    if frequencies.shape[0] != tensor.shape[1] or frequencies.shape[2] * 2 != tensor.shape[3]:
        raise ValueError(
            "RoPE frequencies do not match the token tensor, got "
            f"{tuple(frequencies.shape)} for {tuple(tensor.shape)}."
        )

    complex_tensor = torch.view_as_complex(
        tensor.to(torch.float64).reshape(*tensor.shape[:-1], -1, 2)
    )
    frequencies = frequencies.to(device=tensor.device)
    rotated = torch.view_as_real(complex_tensor * frequencies).flatten(-2)
    return rotated.to(tensor.dtype)
