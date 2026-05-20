# SPDX-License-Identifier: Apache-2.0
"""Error Recycling for Stable-Video-Infinity training.

Implements the per-grid CPU error buffers, timestep binning, distributed
warm-up sync, and the four replacement strategies (random / fifo /
l2_batch / l2_similarity) that match upstream ``train_svi.py``.

Two buffers exist with a slightly confusing upstream naming:
  - ``noise_error_buffer`` stores ``noise_error = x_0_pred - noise_corr_gt``;
    sampled to perturb ``noise`` AND the clean ``latents`` input.
  - ``y_error_buffer``     stores ``y_error     = x_1_pred - latent_corr_gt``;
    sampled to perturb the motion-frame slice of ``y[:, 4:, :num_motion]``.

Both errors are scaled prediction errors at the current timestep — see
``compute_step_errors`` for the algebra.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torch
import torch.distributed as dist

from fastvideo.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

ReplacementStrategy = Literal["random", "fifo", "l2_batch", "l2_similarity"]
_VALID_STRATEGIES: tuple[str, ...] = ("random", "fifo", "l2_batch", "l2_similarity")


class ErrorBuffer:
    """Per-grid bounded buffers of CPU-resident error tensors."""

    def __init__(
        self,
        *,
        num_grids: int,
        max_per_grid: int,
        replacement_strategy: ReplacementStrategy = "random",
    ) -> None:
        if replacement_strategy not in _VALID_STRATEGIES:
            raise ValueError(f"replacement_strategy must be one of {_VALID_STRATEGIES}, got "
                             f"{replacement_strategy!r}")
        self.num_grids = int(num_grids)
        self.max_per_grid = int(max_per_grid)
        self.strategy: ReplacementStrategy = replacement_strategy
        self._grids: dict[int, list[torch.Tensor]] = {i: [] for i in range(self.num_grids)}

    # -- mutate --

    def add(self, tensor_cpu: torch.Tensor, grid_idx: int) -> None:
        bucket = self._grids[grid_idx]
        if len(bucket) < self.max_per_grid:
            bucket.append(tensor_cpu)
            return
        if self.strategy == "random":
            bucket[random.randint(0, len(bucket) - 1)] = tensor_cpu
        elif self.strategy == "fifo":
            bucket.pop(0)
            bucket.append(tensor_cpu)
        elif self.strategy == "l2_batch":
            stack = torch.stack(bucket).flatten(1).float()
            dist_ = torch.norm(stack - tensor_cpu.flatten().unsqueeze(0).float(), p=2, dim=1)
            bucket[int(dist_.argmin())] = tensor_cpu
        elif self.strategy == "l2_similarity":
            best, best_i = float("inf"), -1
            for i, stored in enumerate(bucket):
                d = float((stored.flatten() - tensor_cpu.flatten()).float().norm().item())
                if d < best:
                    best, best_i = d, i
            if best_i >= 0:
                bucket[best_i] = tensor_cpu

    # -- read --

    def sample(
        self,
        grid_idx: int,
        device: torch.device,
        intensity_mod: float = 1.0,
    ) -> torch.Tensor | None:
        bucket = self._grids[grid_idx]
        if not bucket:
            return None
        chosen = random.choice(bucket).to(device)
        if intensity_mod != 1.0:
            chosen = chosen * intensity_mod
        return chosen

    def is_empty_at(self, grid_idx: int) -> bool:
        return not self._grids[grid_idx]

    def total_size(self) -> int:
        return sum(len(b) for b in self._grids.values())

    def size_at(self, grid_idx: int) -> int:
        return len(self._grids[grid_idx])


@dataclass
class GridBinner:
    """Map a timestep value to its nearest reference-grid index."""

    reference_timesteps: torch.Tensor  # [num_grids], CPU float

    def __post_init__(self) -> None:
        self.reference_timesteps = self.reference_timesteps.detach().to("cpu", torch.float32).contiguous()

    def __call__(self, timestep: torch.Tensor | float) -> int:
        val = float(timestep.flatten()[0].item()) if isinstance(timestep, torch.Tensor) else float(timestep)
        val = max(0.0, min(val, 999.0))
        return int((self.reference_timesteps - val).abs().argmin().item())


def build_reference_timesteps(
    *,
    num_grids: int,
    shift: float,
    num_train_timesteps: int = 1000,
    sigma_max: float = 1.0,
    sigma_min: float = 0.003 / 1.002,
) -> torch.Tensor:
    """Mirror diffsynth ``FlowMatchScheduler.get_timesteps`` with ``extra_one_step=True``.

    Linear sigma sweep from ``sigma_max`` down to ``sigma_min`` with one extra
    step trimmed at the tail, then the standard flow-match shift transform.
    Returns timesteps in [0, num_train_timesteps].
    """
    sigmas = torch.linspace(sigma_max, sigma_min, num_grids + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas * num_train_timesteps


def sigma_at(
    scheduler: torch.nn.Module,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Return the scheduler's sigma at the closest discretization to ``timestep``."""
    ts = scheduler.timesteps.to(timestep.device)
    sigmas = scheduler.sigmas.to(timestep.device)
    idx = (ts - timestep.flatten()[0]).abs().argmin()
    return sigmas[idx]


def compute_step_errors(
    *,
    noise_pred: torch.Tensor,
    training_target: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror upstream ``scheduler.step(..., self_corr={True,False})`` for ER updates.

    Upstream computes::

        x_0_pred       = noisy_latents + noise_pred       * (1 - sigma)
        noise_corr_gt  = noisy_latents + training_target  * (1 - sigma)
        noise_error    = x_0_pred - noise_corr_gt = (noise_pred - training_target) * (1 - sigma)

        x_1_pred       = noisy_latents + noise_pred       * (-sigma)
        latent_corr_gt = noisy_latents + training_target  * (-sigma)
        y_error        = x_1_pred - latent_corr_gt = -(noise_pred - training_target) * sigma

    The ``noisy_latents`` term cancels — we only need the prediction delta
    and the current sigma.
    """
    delta = (noise_pred.float() - training_target.float())
    sigma_b = sigma.to(delta.dtype).reshape(*([1] * delta.ndim))
    noise_error = delta * (1.0 - sigma_b)
    y_error = -delta * sigma_b
    return noise_error, y_error


def all_gather_tensor(t: torch.Tensor) -> list[torch.Tensor]:
    """Return ``[world_size]`` per-rank copies; passthrough when not distributed."""
    if not dist.is_available() or not dist.is_initialized():
        return [t]
    world_size = dist.get_world_size()
    if world_size == 1:
        return [t]
    contiguous = t.contiguous()
    gathered = [torch.empty_like(contiguous) for _ in range(world_size)]
    dist.all_gather(gathered, contiguous)
    return gathered


@dataclass
class ErrorRecyclingConfig:
    enabled: bool = False
    error_buffer_k: int = 500
    num_grids: int = 50
    flow_shift: float = 5.0
    buffer_warmup_iter: int = 50
    buffer_replacement_strategy: ReplacementStrategy = "random"
    noise_prob: float = 0.99
    y_prob: float = 0.99
    latent_prob: float = 0.99
    clean_prob: float = 0.1
    clean_buffer_update_prob: float = 0.5
    error_modulate_factor: float = 0.0
    y_error_num: int = 1


class ErrorRecyclingState:
    """Buffers + sampling + warm-up sync for SVI Error Recycling."""

    def __init__(self, cfg: ErrorRecyclingConfig, *, num_train_timesteps: int = 1000) -> None:
        self.cfg = cfg
        self.binner = GridBinner(
            build_reference_timesteps(
                num_grids=cfg.num_grids,
                shift=cfg.flow_shift,
                num_train_timesteps=num_train_timesteps,
            ))
        self.noise_buffer = ErrorBuffer(
            num_grids=cfg.num_grids,
            max_per_grid=cfg.error_buffer_k,
            replacement_strategy=cfg.buffer_replacement_strategy,
        )
        self.y_buffer = ErrorBuffer(
            num_grids=cfg.num_grids,
            max_per_grid=cfg.error_buffer_k,
            replacement_strategy=cfg.buffer_replacement_strategy,
        )

    # -- sampling-time policy --

    def roll(self) -> tuple[bool, bool, bool, bool]:
        """Decide which channels receive error injection this step.

        Returns ``(inject_noise, inject_y, inject_latent, use_clean_input)``.
        ``use_clean_input`` overrides all three injections to False.
        """
        cfg = self.cfg
        inject_noise = random.random() < cfg.noise_prob
        inject_y = random.random() < cfg.y_prob
        inject_latent = random.random() < cfg.latent_prob
        use_clean = random.random() < cfg.clean_prob
        if use_clean:
            inject_noise = inject_y = inject_latent = False
        return inject_noise, inject_y, inject_latent, use_clean

    def sample_noise_error(self, grid_idx: int, like: torch.Tensor) -> torch.Tensor | None:
        return self._sample(self.noise_buffer, grid_idx, like)

    def sample_latent_error(self, grid_idx: int, like: torch.Tensor) -> torch.Tensor | None:
        # Upstream `_sample_latent_error_from_latent_buffer` actually reads y_buffer.
        return self._sample(self.y_buffer, grid_idx, like)

    def sample_y_error(self, grid_idx: int, like: torch.Tensor) -> torch.Tensor | None:
        return self._sample(self.y_buffer, grid_idx, like)

    def _sample(
        self,
        buffer: ErrorBuffer,
        grid_idx: int,
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        if buffer.is_empty_at(grid_idx):
            return None
        f = float(self.cfg.error_modulate_factor)
        mod = random.uniform(1.0 - f, 1.0 + f) if f > 0.0 else 1.0
        return buffer.sample(grid_idx, device=like.device, intensity_mod=mod)

    # -- buffer update (post-loss) --

    def update_local(
        self,
        *,
        noise_error: torch.Tensor,
        y_error: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        grid = self.binner(timestep)
        self.noise_buffer.add(noise_error.detach().to("cpu"), grid)
        self.y_buffer.add(y_error.detach().to("cpu"), grid)

    def update_distributed(
        self,
        *,
        noise_error: torch.Tensor,
        y_error: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        gathered_noise = all_gather_tensor(noise_error)
        gathered_y = all_gather_tensor(y_error)
        gathered_t = all_gather_tensor(timestep.float())
        for ne, ye, ts in zip(gathered_noise, gathered_y, gathered_t, strict=False):
            grid = self.binner(ts)
            self.noise_buffer.add(ne.detach().to("cpu"), grid)
            self.y_buffer.add(ye.detach().to("cpu"), grid)

    def in_warmup(self, iteration: int) -> bool:
        return iteration <= self.cfg.buffer_warmup_iter
