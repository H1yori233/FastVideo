# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo native imports and concise documentation.
"""Euler sampling for a distilled flow map."""

import torch
from torch import Tensor

from fastvideo.models.dits.flux3 import Flux3Transformer2DModel, Packed


def timesteps(steps: int, alpha: float) -> Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def denoise_flow_map(dit: Flux3Transformer2DModel, packed: Packed, schedule: Tensor) -> Tensor:
    targets = schedule[1:].to(packed.data["image"].device)
    for step in range(len(schedule) - 1):
        t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        target = targets[step].reshape(1)
        packed.data["image"] = packed.data["image"] + dt * dit(packed, timesteps_ref=target)
    return packed.data["image"]
