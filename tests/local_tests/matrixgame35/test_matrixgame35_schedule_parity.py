# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.schedule import (
    build_distilled_schedule,
    distilled_noise_seeds,
    x0_renoise_transition,
)


PARITY_SCOPE = "implementation_subcomponent"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_source_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        pytest.skip(f"Pinned upstream source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_released_bf16_schedule_matches_upstream() -> None:
    upstream_schedule = _load_source_module(
        "matrixgame35_upstream_causal_schedule",
        _REPO_ROOT / "Matrix-Game-3.5" / "diffsynth" / "inference" / "causal_schedule.py",
    )
    upstream_flow = _load_source_module(
        "matrixgame35_upstream_flow_match",
        _REPO_ROOT / "Matrix-Game-3.5" / "diffsynth" / "diffusion" / "flow_match.py",
    )
    official_scheduler = upstream_flow.FlowMatchScheduler("Wan")
    expected_timesteps, expected_sigmas = upstream_schedule.prepare_causal_dmd_eval_scheduler(
        official_scheduler,
        (1000, 667, 333),
        model_dtype=torch.bfloat16,
    )
    actual = build_distilled_schedule(model_dtype=torch.bfloat16)

    torch.testing.assert_close(actual.timesteps, expected_timesteps, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.sigmas, expected_sigmas, rtol=0.0, atol=0.0)
    assert actual.model_timesteps.dtype == torch.bfloat16
    assert actual.timesteps.tolist() == [1000.0, 908.0, 712.0]
    assert actual.sigmas.tolist() == [1.0, 0.91015625, 0.71484375]


def test_x0_renoise_matches_upstream_scheduler_transition() -> None:
    upstream_flow = _load_source_module(
        "matrixgame35_upstream_flow_match_transition",
        _REPO_ROOT / "Matrix-Game-3.5" / "diffsynth" / "diffusion" / "flow_match.py",
    )
    scheduler = upstream_flow.FlowMatchScheduler("Wan")
    scheduler.set_timesteps(1000, training=False)
    schedule = build_distilled_schedule(model_dtype=torch.bfloat16)
    scheduler.timesteps = schedule.timesteps.cpu()
    scheduler.sigmas = schedule.sigmas.cpu()

    generator = torch.Generator().manual_seed(35)
    sample = torch.randn(1, 2, 3, 4, generator=generator)
    velocity = torch.randn(1, 2, 3, 4, generator=generator)
    renoise = torch.randn(1, 2, 3, 4, generator=generator)
    x0 = scheduler.step(velocity, schedule.timesteps[0], sample, to_final=True)
    expected = scheduler.add_noise(x0, renoise, schedule.timesteps[1])
    actual = x0_renoise_transition(
        sample,
        velocity,
        schedule.sigmas[0],
        next_sigma=schedule.sigmas[1],
        renoise=renoise,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        x0_renoise_transition(sample, velocity, schedule.sigmas[-1]),
        scheduler.step(velocity, schedule.timesteps[-1], sample, to_final=True),
        rtol=0.0,
        atol=0.0,
    )


def test_released_rng_streams_and_validation() -> None:
    assert distilled_noise_seeds(3407, batch_index=2, chunk_index=6) == (5413, 55413)
    with pytest.raises(ValueError, match="start at 1000"):
        build_distilled_schedule(denoising_ids=(999, 333))
    with pytest.raises(ValueError, match="renoise is required"):
        x0_renoise_transition(torch.zeros(1), torch.zeros(1), 1.0, next_sigma=0.5)
