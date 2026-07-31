# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.configs.pipelines.matrixgame35 import (
    MATRIXGAME35_DISTILLED_PROFILES,
    MatrixGame35DistilledFirstPersonPipelineConfig,
    matrixgame35_distilled_profile_settings,
    resolve_matrixgame35_hiar_scales,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_memory import (
    MatrixGame35DynamicContextEntry,
    MatrixGame35DynamicContextPool,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_profiles import (
    distilled_hiar_noise_seed,
    hiar_sde_corrupt_clean_latents,
    trim_distilled_rolling_latents,
)
from fastvideo.pipelines.basic.matrixgame35.schedule import build_distilled_schedule
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5"))


def _verify_official_ref() -> None:
    if not (OFFICIAL_REF_DIR / "distilled_config.py").is_file():
        pytest.skip("Pinned Matrix-Game 3.5 source is unavailable; set MATRIXGAME35_OFFICIAL_REF_DIR.")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL_REF_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_OFFICIAL_REVISION


def _load_source_module(name: str, relative_path: str) -> ModuleType:
    _verify_official_ref()
    source = OFFICIAL_REF_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned source {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_context_pool():
    _verify_official_ref()
    source = OFFICIAL_REF_DIR / "examples/wanvideo/pipeline/mosaic/causal_memory.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    class_node = next(
        node for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "_CausalKVDynamicContextPool"
    )
    namespace = {
        "normalize_memory_context_selection_policy": lambda value: value,
        "normalize_dynamic_context_selection_policy": lambda value: value,
        "select_pose_near_oldest": lambda candidates, **_kwargs: candidates[0] if candidates else None,
        "_causal_kv_context_pose_score": lambda *_args: 0.0,
        "_stage_generated_context_entries_for_interval": lambda **_kwargs: {},
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_CausalKVDynamicContextPool"]


def _upstream_hiar_seed_expressions() -> dict[bool, ast.expr]:
    _verify_official_ref()
    source = OFFICIAL_REF_DIR / "examples/wanvideo/pipeline/mosaic/causal_rollout.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    expressions: dict[bool, ast.expr] = {}
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "_hiar_sde_corrupt":
            continue
        seed = next((keyword.value for keyword in node.keywords if keyword.arg == "seed"), None)
        if seed is None:
            continue
        constants = {item.value for item in ast.walk(seed) if isinstance(item, ast.Constant)}
        if 7_000_000 in constants:
            expressions[False] = seed
        if 8_000_000 in constants:
            expressions[True] = seed
    assert set(expressions) == {False, True}
    return expressions


def _load_upstream_rollout_function(name: str):
    _verify_official_ref()
    source = OFFICIAL_REF_DIR / "examples/wanvideo/pipeline/mosaic/causal_rollout.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_profile_mapping_matches_pinned_public_config() -> None:
    upstream = _load_source_module("matrixgame35_pinned_distilled_config", "distilled_config.py")

    assert tuple(upstream.PROFILES) == MATRIXGAME35_DISTILLED_PROFILES
    for profile in MATRIXGAME35_DISTILLED_PROFILES:
        config = upstream.DistilledInferenceConfig(profile=profile)
        assert matrixgame35_distilled_profile_settings(profile) == upstream.profile_runtime_settings(config)


def test_hiar_corruption_matches_pinned_schedule_helper() -> None:
    upstream_schedule = _load_source_module(
        "matrixgame35_pinned_hiar_schedule",
        "diffsynth/inference/causal_schedule.py",
    )
    upstream_flow = _load_source_module(
        "matrixgame35_pinned_hiar_flow",
        "diffsynth/diffusion/flow_match.py",
    )
    schedule = build_distilled_schedule(model_dtype=torch.bfloat16)
    scheduler = upstream_flow.FlowMatchScheduler("Wan")
    scheduler.timesteps = schedule.timesteps.cpu()
    scheduler.sigmas = schedule.sigmas.cpu()
    clean = torch.arange(6, dtype=torch.bfloat16).reshape(1, 1, 3, 1, 2)
    noise = torch.linspace(-1, 1, 6, dtype=torch.bfloat16).reshape_as(clean)

    for keep_first_clean in (False, True):
        for scale in (0.0, 0.5, 1.0):
            expected = upstream_schedule.hiar_sde_corrupt_clean_latents(
                scheduler,
                clean,
                schedule.timesteps[1],
                keep_first_clean=keep_first_clean,
                corruption_scale=scale,
                noise=noise,
            )
            actual = hiar_sde_corrupt_clean_latents(
                clean,
                schedule.sigmas[1],
                keep_first_clean=keep_first_clean,
                corruption_scale=scale,
                noise=noise,
            )
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_hiar_seed_derivation_matches_pinned_rollout_expressions() -> None:
    namespace = {
        "args": SimpleNamespace(validation_seed=3407),
        "batch_idx": 2,
        "i": 6,
        "hiar_step_idx": 1,
        "int": int,
    }
    for dynamic_context, expression in _upstream_hiar_seed_expressions().items():
        expected = eval(compile(ast.Expression(expression), "<pinned-hiar-seed>", "eval"), namespace)
        actual = distilled_hiar_noise_seed(
            3407,
            batch_index=2,
            chunk_index=6,
            step_index=1,
            dynamic_context=dynamic_context,
        )
        assert actual == expected


def test_sink_original_anchor_selection_matches_pinned_pool() -> None:
    upstream_pool_cls = _load_upstream_context_pool()
    anchor = torch.full((1, 2, 1, 2, 2), 7.0)
    target = np.repeat(np.eye(4, dtype=np.float32)[None], 12, axis=0)
    upstream = upstream_pool_cls(anchor_position=1, anchor_camera_frame=0, anchor_latent=anchor)
    upstream.force_context_original_anchor = True
    native = MatrixGame35DynamicContextPool(
        original_anchor=MatrixGame35DynamicContextEntry(
            latent=anchor,
            position=1,
            camera_frame=0,
            source_timeline_position=0,
            representative_w2c=np.eye(4, dtype=np.float32),
            source="forced_original_anchor",
        ),
        force_original_anchor=True,
    )

    assert upstream.select_for_chunk(0, target) == []
    assert native.select(target, chunk_index=0, exclude_position=1, exclude_camera_frame=0) is None
    expected = upstream.select_for_chunk(1, target)[0]
    actual = native.select(target, chunk_index=1, exclude_position=1, exclude_camera_frame=0)
    assert actual is not None
    assert actual.source == expected["source"] == "forced_original_anchor"
    assert actual.position == expected["position"] == 1
    assert actual.camera_frame == expected["camera_frame"] == 0
    torch.testing.assert_close(actual.latent, expected["latent"], rtol=0.0, atol=0.0)


def test_profile_config_and_hiar_latent_trim_validation() -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig(
        matrixgame35_distilled_profile="hiar-sde",
        matrixgame35_distilled_hiar_scales=(1.0, 0.5, 0.0),
    )
    assert resolve_matrixgame35_hiar_scales(
        config.matrixgame35_distilled_profile,
        config.matrixgame35_distilled_hiar_scales,
        num_steps=3,
    ) == (1.0, 0.5, 0.0)
    latents = torch.arange(22, dtype=torch.float32).reshape(1, 1, 22, 1, 1)
    trimmed = trim_distilled_rolling_latents(latents, frames_per_chunk=3, window_chunks=7)
    expected = _load_upstream_rollout_function("_causal_kv_trim_rolling_latents")(
        latents,
        frames_per_chunk=3,
        window_chunks=7,
    )
    torch.testing.assert_close(trimmed, expected, rtol=0.0, atol=0.0)
    assert trimmed.flatten().tolist() == [3.0, *[float(value) for value in range(4, 22)]]

    with pytest.raises(ValueError, match="only valid"):
        MatrixGame35DistilledFirstPersonPipelineConfig(matrixgame35_distilled_hiar_scales=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="one value per"):
        MatrixGame35DistilledFirstPersonPipelineConfig(
            matrixgame35_distilled_profile="hiar-sde",
            matrixgame35_distilled_hiar_scales=(1.0, 0.5),
        )
