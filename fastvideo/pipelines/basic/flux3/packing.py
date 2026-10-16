# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo native imports and concise documentation.
"""FLUX3 mode-major packing and image position scaling."""

from typing import cast

import torch
from torch import Tensor

from fastvideo.models.dits.flux3 import IN_CHANNELS, MODES, Packed

IMAGE_ROPE_SCALE = 0.5


def pack(samples: list[list[tuple[Tensor, Tensor] | None]]) -> Packed:
    """samples[s] = [text, image_cond or None, image], each a (tokens, ids) pair."""
    data: dict[str, list[Tensor]] = {m: [] for m in MODES}
    ids: dict[str, list[Tensor]] = {m: [] for m in MODES}
    mode_indices: dict[str, list[tuple[int, int]]] = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))
    device = cast(tuple[Tensor, Tensor], samples[0][0])[0].device

    def cat(parts: list[Tensor], width: int, dtype: torch.dtype) -> Tensor:
        return torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dtype)

    data_tensors = {m: cat(data[m], IN_CHANNELS[m], torch.bfloat16) for m in MODES}
    id_tensors = {m: cat(ids[m], 4, torch.float32) for m in MODES}
    return Packed(
        data=data_tensors,
        ids=id_tensors,
        timesteps={m: torch.zeros(data_tensors[m].shape[0], device=device, dtype=torch.bfloat16)
                   for m in MODES},
        mode_indices=mode_indices,
    )


def scaled_ids(ids: Tensor) -> Tensor:
    ids = ids.float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids
