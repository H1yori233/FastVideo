# SPDX-License-Identifier: Apache-2.0
"""Released Matrix-Game 3.5 distilled schedule and transition contract."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


RELEASED_DENOISING_IDS = (1000, 667, 333)


@dataclass(frozen=True)
class MatrixGame35DistilledSchedule:
    """Few-step values after the official BF16 quantization boundary."""

    denoising_ids: tuple[int, ...]
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    model_timesteps: torch.Tensor


def _validate_denoising_ids(values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in values)
    if len(values) not in (2, 3, 4):
        raise ValueError(f"distilled denoising schedule must contain 2, 3, or 4 steps, got {values!r}.")
    if values[0] != 1000:
        raise ValueError(f"distilled denoising schedule must start at 1000, got {values!r}.")
    if any(value <= 0 or value > 1000 for value in values):
        raise ValueError(f"distilled denoising ids must lie in [1, 1000], got {values!r}.")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError(f"distilled denoising ids must be strictly descending, got {values!r}.")
    return values


def build_distilled_schedule(
    *,
    denoising_ids: Sequence[int] = RELEASED_DENOISING_IDS,
    device: torch.device | str = "cpu",
    model_dtype: torch.dtype = torch.bfloat16,
) -> MatrixGame35DistilledSchedule:
    """Resolve CF++ ids on the 1000-row Wan shift-5 schedule."""
    ids = _validate_denoising_ids(denoising_ids)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    if len(scheduler.timesteps) != 1000 or len(scheduler.sigmas) != 1000:
        raise RuntimeError("Matrix-Game 3.5 requires a 1000-row Wan scheduler table.")

    indices = torch.as_tensor([1000 - value for value in ids], dtype=torch.long)
    raw_timesteps = scheduler.timesteps.index_select(0, indices)
    raw_sigmas = scheduler.sigmas.index_select(0, indices)
    model_timesteps = raw_timesteps.to(device=device, dtype=model_dtype)
    # The upstream runtime quantizes both tables to the model dtype and then
    # returns FP32 values to its scheduler arithmetic.
    timesteps = model_timesteps.float()
    sigmas = raw_sigmas.to(device=device, dtype=model_dtype).float()
    return MatrixGame35DistilledSchedule(ids, timesteps, sigmas, model_timesteps)


def x0_renoise_transition(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    next_sigma: torch.Tensor | float | None = None,
    renoise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the released ``x0_renoise`` transition, not an ODE step."""
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    x0 = sample - sigma_tensor * velocity
    if next_sigma is None:
        return x0
    if renoise is None:
        raise ValueError("renoise is required when next_sigma is provided.")
    if renoise.shape != sample.shape:
        raise ValueError(f"renoise shape must match sample, got {tuple(renoise.shape)} vs {tuple(sample.shape)}.")
    next_sigma_tensor = torch.as_tensor(next_sigma, device=sample.device, dtype=sample.dtype)
    return (1.0 - next_sigma_tensor) * x0 + next_sigma_tensor * renoise.to(
        device=sample.device, dtype=sample.dtype
    )


def distilled_noise_seeds(base_seed: int, *, batch_index: int, chunk_index: int) -> tuple[int, int]:
    """Return the official initial-noise and independent re-noise streams."""
    initial = int(base_seed) + int(batch_index) * 1000 + int(chunk_index)
    return initial, initial + 50000
