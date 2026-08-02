# SPDX-License-Identifier: Apache-2.0
"""Real-weight parity for the distilled Matrix-Game 3.5 transformer.

Coverage scope: both. The official side strict-loads the published 825-tensor
BF16 checkpoint into the pinned ``WanModel``. The FastVideo side strict-loads
the converted transformer through ``TransformerLoader`` with causal execution
enabled and subject-reference memory disabled.

The CUDA path runs the models sequentially and compares the causal lifecycle at
4x4 latent resolution: anchor cache bootstrap, read-only denoise, final cache
write, and the released 19-frame rolling-window trim. Missing source, weights,
converted weights, or CUDA are scaffold skips and are not parity evidence.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pytest
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
import torch
from torch.testing import assert_close

from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.basic.matrixgame35.causal_kv_cache import (
    concat_causal_kv_caches,
    copy_causal_kv_caches,
    tail_causal_kv_cache_frames,
    trim_causal_kv_rolling_window,
)
from scripts.checkpoint_conversion import matrixgame35_to_diffusers as converter
from tests.local_tests.matrixgame35._upstream import (
    PINNED_OFFICIAL_REVISION,
    UpstreamTransformerModules,
    load_upstream_pipeline,
    load_upstream_transformer,
)


os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29536")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_SCOPE = "both"

OFFICIAL_HF_REPO = "RiemannDynamics/Matrix-Game-3.5-Distilled"
OFFICIAL_HF_REVISION = "0b38ca0b0dda2bb994c570e183ad36d1acd53be2"
OFFICIAL_WEIGHT_NAME = "first-person.safetensors"
OFFICIAL_WEIGHT_BYTES = 9_999_659_704
OFFICIAL_WEIGHT_SHA256 = "de476e7fc0bdd756aafb101a2b80040f65b3ad62dafea109e299aafa599b8094"
EXPECTED_KEY_COUNT = 825
DISTILLED_VARIANT = converter.VARIANTS["distilled_first_person"]

OFFICIAL_REF_DIR = Path(
    os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5")
)
DISTILLED_WEIGHTS_DIR = Path(
    os.getenv(
        "MATRIXGAME35_DISTILLED_WEIGHTS_DIR",
        REPO_ROOT / "official_weights" / "matrixgame35" / "distilled",
    )
)
OFFICIAL_WEIGHT_PATH = Path(
    os.getenv(
        "MATRIXGAME35_DISTILLED_FIRST_PERSON_WEIGHTS",
        DISTILLED_WEIGHTS_DIR / OFFICIAL_WEIGHT_NAME,
    )
)
CONVERTED_TRANSFORMER_DIR = Path(
    os.getenv(
        "MATRIXGAME35_DISTILLED_CONVERTED_TRANSFORMER_DIR",
        REPO_ROOT
        / "converted_weights"
        / "matrixgame35"
        / "distilled_first_person"
        / "transformer",
    )
)

OFFICIAL_MODEL_KWARGS: dict[str, Any] = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "use_prope": True,
    "prope_disable_native_rope": False,
    "prope_disable_t_rope": False,
    "prope_camera_layout": "full",
    "subject_ref_memory_enabled": False,
}

DISTILLED_CONFIG_CONTRACT: dict[str, Any] = {
    "_class_name": "MatrixGame35Transformer3DModel",
    "patch_size": [1, 2, 2],
    "in_channels": 48,
    "out_channels": 48,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "ffn_dim": 14336,
    "num_layers": 30,
    "text_dim": 4096,
    "freq_dim": 256,
    "use_prope": True,
    "prope_attention_interval": 1,
    "prope_camera_layout": "full",
    "prope_disable_native_rope": False,
    "subject_ref_memory_max_refs": 0,
    "causal": True,
    "causal_chunk_size": 3,
    "causal_window_size": 21,
}


@dataclass(frozen=True)
class CacheHelpers:
    copy: Callable[..., list[dict[str, Any]]]
    concat: Callable[..., list[dict[str, Any]]]
    tail: Callable[..., list[dict[str, Any]]]
    trim: Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class LifecycleInputs:
    anchor: torch.Tensor
    noisy: torch.Tensor
    clean: torch.Tensor
    context: torch.Tensor
    camera_info: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _skip_if_reference_missing() -> None:
    source = OFFICIAL_REF_DIR / "diffsynth" / "models" / "wan_video_dit.py"
    if not source.is_file():
        pytest.skip(
            "Pinned Matrix-Game 3.5 reference is absent; set "
            f"MATRIXGAME35_OFFICIAL_REF_DIR to commit {PINNED_OFFICIAL_REVISION}."
        )


def _skip_if_official_weight_missing() -> None:
    if not OFFICIAL_WEIGHT_PATH.is_file():
        pytest.skip(
            f"Official distilled transformer weight is absent: {OFFICIAL_WEIGHT_PATH}. "
            f"Stage {OFFICIAL_HF_REPO}@{OFFICIAL_HF_REVISION}/{OFFICIAL_WEIGHT_NAME} "
            "or set MATRIXGAME35_DISTILLED_FIRST_PERSON_WEIGHTS."
        )


def _converted_weight_paths() -> list[Path]:
    return sorted(CONVERTED_TRANSFORMER_DIR.glob("*.safetensors"))


def _skip_if_converted_weights_missing() -> None:
    if not CONVERTED_TRANSFORMER_DIR.is_dir() or not _converted_weight_paths():
        pytest.skip(
            "Converted distilled transformer is absent; set "
            "MATRIXGAME35_DISTILLED_CONVERTED_TRANSFORMER_DIR "
            f"(expected {CONVERTED_TRANSFORMER_DIR})."
        )


@lru_cache(maxsize=1)
def _official_checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    with OFFICIAL_WEIGHT_PATH.open("rb") as checkpoint:
        while chunk := checkpoint.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_header(
    paths: list[Path],
    *,
    official_namespace: bool,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    raw_shapes: dict[str, tuple[int, ...]] = {}
    raw_dtypes: dict[str, str] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                assert key not in raw_shapes, f"Duplicate checkpoint tensor across shards: {key}"
                tensor_slice = checkpoint.get_slice(key)
                raw_shapes[key] = tuple(tensor_slice.get_shape())
                raw_dtypes[key] = str(tensor_slice.get_dtype())

    if not official_namespace:
        return raw_shapes, raw_dtypes
    _, normalized_names = converter.normalize_namespace(list(raw_shapes))
    return (
        {normalized_names[key]: shape for key, shape in raw_shapes.items()},
        {normalized_names[key]: dtype for key, dtype in raw_dtypes.items()},
    )


def _assert_distilled_config(config: dict[str, Any]) -> None:
    for key, expected in DISTILLED_CONFIG_CONTRACT.items():
        assert config.get(key) == expected, (
            f"Unexpected distilled transformer config {key}={config.get(key)!r}; "
            f"expected {expected!r}"
        )


def _build_official_meta_model(
    modules: UpstreamTransformerModules,
) -> torch.nn.Module:
    with torch.device("meta"):
        model = modules.wan_video_dit.WanModel(**OFFICIAL_MODEL_KWARGS)
    assert model.use_prope is True
    assert model.seperated_timestep is True
    assert model.subject_ref_memory_enabled is False
    assert not hasattr(model, "subject_ref_index_embedding")
    assert len(model.blocks) == 30
    assert len(model.state_dict()) == EXPECTED_KEY_COUNT
    return model


def _load_official_model(
    modules: UpstreamTransformerModules,
    device: torch.device,
) -> torch.nn.Module:
    model = _build_official_meta_model(modules)
    raw_state = safetensors_load_file(str(OFFICIAL_WEIGHT_PATH), device="cpu")
    _, normalized_names = converter.normalize_namespace(list(raw_state))
    state = {normalized_names[key]: tensor for key, tensor in raw_state.items()}
    del raw_state

    expected = model.state_dict()
    assert set(state) == set(expected), (
        f"Official strict key mismatch: missing={sorted(set(expected) - set(state))[:8]} "
        f"unexpected={sorted(set(state) - set(expected))[:8]}"
    )
    shape_mismatches = {
        key: (tuple(state[key].shape), tuple(expected[key].shape))
        for key in state
        if tuple(state[key].shape) != tuple(expected[key].shape)
    }
    assert not shape_mismatches, f"Official checkpoint shape mismatch: {shape_mismatches}"
    assert {tensor.dtype for tensor in state.values()} == {torch.bfloat16}

    incompatible = model.load_state_dict(state, strict=True, assign=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    del state

    model.freqs = modules.wan_video_dit.precompute_freqs_cis_3d(128)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    assert not any(parameter.is_meta for parameter in model.parameters())
    return model


def _load_fastvideo_model(device: torch.device) -> torch.nn.Module:
    config_path = CONVERTED_TRANSFORMER_DIR / "config.json"
    if not config_path.is_file():
        pytest.fail(f"Converted transformer directory has no config.json: {config_path}")
    _assert_distilled_config(json.loads(config_path.read_text(encoding="utf-8")))

    from fastvideo.configs.models.dits.matrixgame35 import MatrixGame35WanVideoConfig
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.matrixgame35 import MatrixGame35Transformer3DModel
    from fastvideo.models.loader.component_loader import TransformerLoader

    args = FastVideoArgs(
        model_path=str(CONVERTED_TRANSFORMER_DIR),
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=MatrixGame35WanVideoConfig(),
            dit_precision="bf16",
        ),
    )
    model = TransformerLoader().load(str(CONVERTED_TRANSFORMER_DIR), args)
    assert isinstance(model, MatrixGame35Transformer3DModel)
    assert args.model_paths["transformer"] == str(CONVERTED_TRANSFORMER_DIR)
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert next(model.parameters()).device == device
    assert next(model.parameters()).dtype == torch.bfloat16
    assert model.causal is True
    assert model.subject_ref_memory_enabled is False
    assert not hasattr(model, "subject_ref_index_embedding")
    assert len(model.state_dict()) == EXPECTED_KEY_COUNT
    return model.eval()


def _make_lifecycle_inputs() -> LifecycleInputs:
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    anchor = torch.randn(1, 48, 1, 4, 4, generator=generator)
    noisy = torch.randn(1, 48, 3, 4, 4, generator=generator)
    clean = torch.randn(1, 48, 3, 4, 4, generator=generator)
    context = torch.randn(1, 16, 4096, generator=generator)

    camera_frames = 22
    projection = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, camera_frames, 4, 1, 1)
    for frame in range(camera_frames):
        for subframe in range(4):
            projection[0, frame, subframe, 0, 0] = 1.0 + 0.002 * frame
            projection[0, frame, subframe, 1, 1] = 1.0 + 0.003 * subframe
            projection[0, frame, subframe, 0, 3] = 0.001 * (frame + subframe)
            projection[0, frame, subframe, 2, 3] = -0.001 * frame
    projection_transpose = projection.transpose(-1, -2).contiguous()
    projection_inverse = torch.linalg.inv(projection)
    camera_info = (
        projection.clone(),
        (projection, projection_transpose, projection_inverse),
    )
    return LifecycleInputs(anchor, noisy, clean, context, camera_info)


def _to_device(
    inputs: LifecycleInputs,
    device: torch.device,
) -> LifecycleInputs:
    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, dtype=torch.bfloat16)

    return LifecycleInputs(
        anchor=move(inputs.anchor),
        noisy=move(inputs.noisy),
        clean=move(inputs.clean),
        context=move(inputs.context),
        camera_info=(
            move(inputs.camera_info[0]),
            tuple(move(matrix) for matrix in inputs.camera_info[1]),
        ),
    )


def _load_official_cache_helpers() -> CacheHelpers:
    source = OFFICIAL_REF_DIR / "examples" / "wanvideo" / "pipeline" / "mosaic" / "causal_rollout.py"
    if not source.is_file():
        pytest.fail(f"Pinned upstream causal rollout is missing: {source}")
    names = {
        "_causal_kv_cache_copy",
        "_causal_kv_concat_caches",
        "_causal_kv_frame_count",
        "_causal_kv_slice_cache_frames",
        "_causal_kv_tail_cache_frames",
        "_causal_kv_trim_rolling_window",
    }
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return CacheHelpers(
        copy=namespace["_causal_kv_cache_copy"],
        concat=namespace["_causal_kv_concat_caches"],
        tail=namespace["_causal_kv_tail_cache_frames"],
        trim=namespace["_causal_kv_trim_rolling_window"],
    )


def _native_cache_helpers() -> CacheHelpers:
    return CacheHelpers(
        copy=copy_causal_kv_caches,
        concat=concat_causal_kv_caches,
        tail=tail_causal_kv_cache_frames,
        trim=trim_causal_kv_rolling_window,
    )


def _clone_caches_to_cpu(caches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "k": None if cache["k"] is None else cache["k"].detach().cpu().clone(),
            "v": None if cache["v"] is None else cache["v"].detach().cpu().clone(),
            "positions": list(cache["positions"]),
            "frames": list(cache["frames"]),
            "chunk_ids": list(cache["chunk_ids"]),
        }
        for cache in caches
    ]


def _assert_caches_exact(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    assert len(actual) == len(expected)
    for actual_cache, expected_cache in zip(actual, expected, strict=True):
        for name in ("positions", "frames", "chunk_ids"):
            assert actual_cache[name] == expected_cache[name]
        for name in ("k", "v"):
            if expected_cache[name] is None:
                assert actual_cache[name] is None
            else:
                assert torch.equal(actual_cache[name].detach().cpu(), expected_cache[name])


def _prefill_released_rolling_window(caches: list[dict[str, Any]]) -> None:
    """Expand the real anchor write to the released anchor + six-chunk window."""

    for cache in caches:
        assert cache["k"] is not None and cache["v"] is not None
        assert cache["k"].shape[1] == cache["v"].shape[1] == 4
        cache["k"] = cache["k"].repeat(1, 19, 1).contiguous()
        cache["v"] = cache["v"].repeat(1, 19, 1).contiguous()
        cache["positions"] = list(range(1, 20))
        cache["frames"] = list(range(19))
        cache["chunk_ids"] = [-1] * 19


def _execute_causal_lifecycle(
    *,
    forward: Callable[..., torch.Tensor],
    init_caches: Callable[[], list[dict[str, Any]]],
    helpers: CacheHelpers,
    inputs: LifecycleInputs,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    caches = init_caches()
    assert len(caches) == 30

    bootstrap = forward(
        inputs.anchor,
        torch.zeros(1, device=inputs.anchor.device, dtype=torch.bfloat16),
        caches,
        current_positions=[1],
        current_frames=[0],
        cache_positions=None,
        cache_frames=None,
        cache_read_chunk_id=None,
        current_cache_chunk_ids=[-1],
        write_cache=True,
    )
    assert all(cache["positions"] == [1] and cache["frames"] == [0] for cache in caches)

    _prefill_released_rolling_window(caches)
    rolling_cache = helpers.copy(caches)
    read_cache = helpers.copy(caches)
    cache_before_denoise = _clone_caches_to_cpu(read_cache)

    denoise = forward(
        inputs.noisy,
        torch.full((3,), 1000.0, device=inputs.noisy.device, dtype=torch.bfloat16),
        read_cache,
        current_positions=[20, 21, 22],
        current_frames=[19, 20, 21],
        cache_positions=list(range(1, 20)),
        cache_frames=list(range(19)),
        cache_read_chunk_id=6,
        current_cache_chunk_ids=None,
        write_cache=False,
    )
    _assert_caches_exact(read_cache, cache_before_denoise)

    final = forward(
        inputs.clean,
        torch.zeros(3, device=inputs.clean.device, dtype=torch.bfloat16),
        read_cache,
        current_positions=[20, 21, 22],
        current_frames=[19, 20, 21],
        cache_positions=list(range(1, 20)),
        cache_frames=list(range(19)),
        cache_read_chunk_id=6,
        current_cache_chunk_ids=None,
        write_cache=True,
    )
    assert all(cache["positions"] == list(range(1, 23)) for cache in read_cache)
    assert all(cache["frames"] == list(range(22)) for cache in read_cache)

    current_cache = helpers.tail(read_cache, 3, context="distilled real-weight parity")
    trimmed = helpers.trim(
        helpers.concat(rolling_cache, current_cache),
        frames_per_chunk=3,
        window_chunks=7,
    )
    assert all(cache["positions"] == list(range(4, 23)) for cache in trimmed)
    assert all(cache["frames"] == list(range(3, 22)) for cache in trimmed)
    assert all(cache["chunk_ids"] == [-1] * 19 for cache in trimmed)

    outputs = {
        "bootstrap": bootstrap.detach().float().cpu(),
        "read_only_denoise": denoise.detach().float().cpu(),
        "final_write": final.detach().float().cpu(),
    }
    return outputs, _clone_caches_to_cpu(trimmed)


def _run_official_lifecycle(
    pipeline: Any,
    model: torch.nn.Module,
    inputs: LifecycleInputs,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    def forward(
        latents: torch.Tensor,
        timestep: torch.Tensor,
        caches: list[dict[str, Any]],
        **kwargs: Any,
    ) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return pipeline.model_fn_causal_kv(
                model,
                latents_chunk=latents,
                timestep_frames=timestep,
                context=inputs.context,
                camera_info=inputs.camera_info,
                cur_positions=kwargs.pop("current_positions"),
                cur_frames=kwargs.pop("current_frames"),
                caches=caches,
                cur_cache_chunk_ids=kwargs.pop("current_cache_chunk_ids"),
                **kwargs,
            )

    return _execute_causal_lifecycle(
        forward=forward,
        init_caches=lambda: pipeline.init_causal_kv_caches(30),
        helpers=_load_official_cache_helpers(),
        inputs=inputs,
    )


def _run_fastvideo_lifecycle(
    model: torch.nn.Module,
    inputs: LifecycleInputs,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    def forward(
        latents: torch.Tensor,
        timestep: torch.Tensor,
        caches: list[dict[str, Any]],
        **kwargs: Any,
    ) -> torch.Tensor:
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            return model(
                hidden_states=latents,
                encoder_hidden_states=inputs.context,
                timestep=timestep,
                camera_info=inputs.camera_info,
                kv_caches=caches,
                **kwargs,
            )

    return _execute_causal_lifecycle(
        forward=forward,
        init_caches=model.init_causal_kv_caches,
        helpers=_native_cache_helpers(),
        inputs=inputs,
    )


def _assert_tensor_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    atol: float = 0.1,
) -> None:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference = (actual - expected).abs()
    expected_abs_mean = expected.abs().mean().clamp_min(1e-6)
    actual_abs_mean = actual.abs().mean()
    abs_mean_drift = (actual_abs_mean - expected_abs_mean).abs() / expected_abs_mean
    normalized_mean_error = difference.mean() / expected_abs_mean
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    print(
        f"Matrix-Game 3.5 distilled {label}: "
        f"official_abs_mean={expected_abs_mean.item():.6f} "
        f"fastvideo_abs_mean={actual_abs_mean.item():.6f} "
        f"diff_max={difference.max().item():.6f} "
        f"diff_mean={difference.mean().item():.6f} "
        f"normalized_mean_error={normalized_mean_error.item():.6f} "
        f"abs_mean_drift={abs_mean_drift.item():.6f} "
        f"cosine={cosine.item():.6f}"
    )
    assert abs_mean_drift < 0.05
    assert normalized_mean_error < 0.05
    assert cosine > 0.99
    assert_close(actual, expected, atol=atol, rtol=0.1)


def _assert_cache_parity(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    assert len(actual) == len(expected) == 30
    for block_index, (actual_cache, expected_cache) in enumerate(
        zip(actual, expected, strict=True)
    ):
        for name in ("positions", "frames", "chunk_ids"):
            assert actual_cache[name] == expected_cache[name]
        for name in ("k", "v"):
            assert actual_cache[name] is not None and expected_cache[name] is not None
            # The 19-frame rolling cache accumulates cross-kernel BF16 rounding;
            # primary outputs retain the tighter 0.1 bound and every cache still
            # has independent drift, normalized-error, and cosine gates.
            _assert_tensor_parity(
                actual_cache[name].float(),
                expected_cache[name].float(),
                label=f"trimmed_cache.block_{block_index}.{name}",
                atol=0.125,
            )


def test_distilled_published_contract_without_assets() -> None:
    """Freeze the published checkpoint identity and asset-free 825-key surface."""

    assert DISTILLED_VARIANT.hf_repo == OFFICIAL_HF_REPO
    assert DISTILLED_VARIANT.revision == OFFICIAL_HF_REVISION
    assert DISTILLED_VARIANT.published_filename == OFFICIAL_WEIGHT_NAME
    assert DISTILLED_VARIANT.published_bytes == OFFICIAL_WEIGHT_BYTES
    assert DISTILLED_VARIANT.published_sha256 == OFFICIAL_WEIGHT_SHA256
    assert DISTILLED_VARIANT.subject_ref_memory_max_refs == 0
    assert DISTILLED_VARIANT.causal is True

    expected_shapes = converter.build_expected_official_shapes(DISTILLED_VARIANT)
    assert len(expected_shapes) == EXPECTED_KEY_COUNT
    assert not set(converter.SUBJECT_REF_KEYS) & set(expected_shapes)
    _assert_distilled_config(converter.build_transformer_config(DISTILLED_VARIANT))


def test_distilled_official_checkpoint_header_and_sha256() -> None:
    """Validate the complete published file before allocating transformer weights."""

    _skip_if_official_weight_missing()
    assert OFFICIAL_WEIGHT_PATH.stat().st_size == OFFICIAL_WEIGHT_BYTES
    assert _official_checkpoint_sha256() == OFFICIAL_WEIGHT_SHA256

    shapes, dtypes = _normalized_header([OFFICIAL_WEIGHT_PATH], official_namespace=True)
    assert shapes == converter.build_expected_official_shapes(DISTILLED_VARIANT)
    assert len(shapes) == EXPECTED_KEY_COUNT
    assert set(dtypes.values()) == {"BF16"}
    assert not set(converter.SUBJECT_REF_KEYS) & set(shapes)


def test_distilled_converted_checkpoint_header_and_config() -> None:
    """Require a complete causal, subject-free converted transformer directory."""

    _skip_if_converted_weights_missing()
    config_path = CONVERTED_TRANSFORMER_DIR / "config.json"
    assert config_path.is_file(), f"Converted transformer has no config.json: {config_path}"
    _assert_distilled_config(json.loads(config_path.read_text(encoding="utf-8")))

    shapes, dtypes = _normalized_header(_converted_weight_paths(), official_namespace=False)
    expected_shapes = {
        converter._mapped_key(key): shape
        for key, shape in converter.build_expected_official_shapes(DISTILLED_VARIANT).items()
    }
    assert shapes == expected_shapes
    assert len(shapes) == EXPECTED_KEY_COUNT
    assert set(dtypes.values()) == {"BF16"}
    assert not any("subject_ref" in key for key in shapes)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Real-weight Matrix-Game 3.5 distilled transformer parity requires CUDA.",
)
def test_distilled_real_weight_causal_kv_lifecycle_parity() -> None:
    """Strict-load and compare the real causal transformer and rolling K/V state."""

    assert torch.cuda.is_bf16_supported(), "Distilled parity requires BF16-capable CUDA hardware"
    _skip_if_reference_missing()
    _skip_if_official_weight_missing()
    assert OFFICIAL_WEIGHT_PATH.stat().st_size == OFFICIAL_WEIGHT_BYTES
    assert _official_checkpoint_sha256() == OFFICIAL_WEIGHT_SHA256
    _skip_if_converted_weights_missing()

    modules = load_upstream_transformer(OFFICIAL_REF_DIR)
    pipeline = load_upstream_pipeline(OFFICIAL_REF_DIR)
    device = torch.device("cuda:0")
    cpu_inputs = _make_lifecycle_inputs()

    official_inputs = _to_device(cpu_inputs, device)
    official = _load_official_model(modules, device)
    official_outputs, official_cache = _run_official_lifecycle(
        pipeline,
        official,
        official_inputs,
    )
    del official_inputs
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo_inputs = _to_device(cpu_inputs, device)
    fastvideo = _load_fastvideo_model(device)
    fastvideo_outputs, fastvideo_cache = _run_fastvideo_lifecycle(
        fastvideo,
        fastvideo_inputs,
    )
    del fastvideo_inputs
    del fastvideo
    gc.collect()
    torch.cuda.empty_cache()

    assert set(fastvideo_outputs) == set(official_outputs)
    for name in official_outputs:
        _assert_tensor_parity(
            fastvideo_outputs[name],
            official_outputs[name],
            label=name,
        )
    _assert_cache_parity(fastvideo_cache, official_cache)
