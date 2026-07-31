# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.pipelines.matrixgame35 import MatrixGame35DistilledFirstPersonPipelineConfig
from fastvideo.pipelines.basic.matrixgame35 import distilled_standard_stages
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_kv import (
    concat_causal_kv_caches,
    trim_causal_kv_rolling_window,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_memory import (
    DA3_PROCESS_RES,
    MatrixGame35DistilledDepthAnything3Adapter,
    MatrixGame35DistilledMemoryResult,
    MatrixGame35DistilledPatchMemory,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_pipeline import (
    MatrixGame35DistilledFirstPersonPipeline,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_stages import (
    DISTILLED_NEGATIVE_PROMPT,
    MatrixGame35DistilledInputValidationStage,
    MatrixGame35DistilledRolloutStage,
)
from fastvideo.pipelines.basic.matrixgame35.prompts import MatrixGame35TextEncodingStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"


def _load_official_cache_helpers():
    source = _OFFICIAL_DIR / "examples/wanvideo/pipeline/mosaic/causal_rollout.py"
    assert source.is_file(), f"Pinned upstream causal rollout is missing: {source}"
    revision = subprocess.run(
        ["git", "-C", str(_OFFICIAL_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_OFFICIAL_REVISION
    names = {
        "_causal_kv_cache_copy",
        "_causal_kv_concat_caches",
        "_causal_kv_frame_count",
        "_causal_kv_slice_cache_frames",
        "_causal_kv_tail_cache_frames",
        "_causal_kv_trim_rolling_window",
    }
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _cache(frames: list[int], positions: list[int], values: torch.Tensor) -> list[dict]:
    return [{
        "k": values.clone(),
        "v": values.clone() + 0.25,
        "positions": positions,
        "frames": frames,
        "chunk_ids": [-1] * len(frames),
    }]


def _assert_cache_equal(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for actual_block, expected_block in zip(actual, expected):
        assert actual_block["positions"] == expected_block["positions"]
        assert actual_block["frames"] == expected_block["frames"]
        assert actual_block["chunk_ids"] == expected_block["chunk_ids"]
        torch.testing.assert_close(actual_block["k"], expected_block["k"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_block["v"], expected_block["v"], rtol=0.0, atol=0.0)


def test_rolling_cache_matches_pinned_upstream_anchor_and_eviction() -> None:
    official = _load_official_cache_helpers()
    values = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
    native = _cache([0], [1], values)
    expected = _cache([0], [1], values)

    for chunk_index in range(10):
        frames = [1 + chunk_index * 3 + offset for offset in range(3)]
        positions = [2 + chunk_index * 3 + offset for offset in range(3)]
        chunk_values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + 100 * chunk_index
        chunk = _cache(frames, positions, chunk_values)
        native = trim_causal_kv_rolling_window(
            concat_causal_kv_caches(native, chunk),
            frames_per_chunk=3,
            window_chunks=7,
        )
        expected = official["_causal_kv_trim_rolling_window"](
            official["_causal_kv_concat_caches"](expected, chunk),
            frames_per_chunk=3,
            window_chunks=7,
        )
        _assert_cache_equal(native, expected)
        assert len(native[0]["frames"]) <= 19

    assert native[0]["frames"] == [12, *range(13, 31)]
    assert native[0]["positions"] == [13, *range(14, 32)]


def test_distilled_memory_uses_coverage_far_zbuffer_and_preserves_holes() -> None:
    memory = MatrixGame35DistilledPatchMemory()
    frame_count = 6
    memory.append(
        latents=torch.stack([torch.full((1, 4, 4), float(index + 1)) for index in range(frame_count)], dim=1),
        w2c=np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0),
        depths=np.ones((frame_count, 64, 64), dtype=np.float32),
    )
    query_w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 12, axis=0)
    query_K = np.repeat(np.eye(3, dtype=np.float32)[None], 12, axis=0)
    result = memory.query(
        anchor_w2c=np.eye(4, dtype=np.float32),
        query_w2c=query_w2c,
        query_intrinsics=query_K,
    )

    assert result.candidate_frame_ids == ((0, 2, 4, 4, 4),) * 3
    assert result.latents.shape == (1, 3, 4, 4)
    assert bool(result.valid_mask.all())
    torch.testing.assert_close(result.latents, torch.ones_like(result.latents))

    off_canvas = query_w2c.copy()
    off_canvas[:, 0, 3] = 1000.0
    holes = memory.query(
        anchor_w2c=np.eye(4, dtype=np.float32),
        query_w2c=off_canvas,
        query_intrinsics=query_K,
    )
    assert not bool(holes.valid_mask.any())
    assert not bool(holes.latents.any())


class _FakeDA3:

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def eval(self):
        return self

    def to(self, _device):
        return self

    def inference(self, frames, **kwargs):
        self.calls.append({"count": len(frames), **kwargs})
        height, width = np.asarray(frames[0]).shape[:2]
        return SimpleNamespace(depth=np.ones((len(frames), height, width), dtype=np.float32))


def test_da3_boundary_is_explicit_and_uses_released_448_bf16_contract() -> None:
    estimator = _FakeDA3()
    adapter = MatrixGame35DistilledDepthAnything3Adapter(
        "/pinned/da3",
        device="cpu",
        estimator=estimator,
    )
    depths = adapter.estimate_depth([np.zeros((8, 12, 3), dtype=np.uint8)] * 2)

    assert depths.shape == (2, 8, 12)
    assert estimator.calls == [{"count": 2, "use_ray_pose": False, "process_res": DA3_PROCESS_RES}]


class _Posterior:

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _FakeVAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
            scale_factor_spatial=16,
        )
        self.z_dim = 2
        self.encode_shapes: list[tuple[int, ...]] = []
        self.decode_lengths: list[int] = []
        self.public_decode_calls = 0
        self.to_calls: list[object] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)

    def encode(self, video: torch.Tensor) -> _Posterior:
        self.encode_shapes.append(tuple(video.shape))
        spatial = F.avg_pool3d(video[:, :2], kernel_size=(1, 16, 16))
        if spatial.shape[2] == 5:
            spatial = torch.stack((spatial[:, :, 0], spatial[:, :, -1]), dim=2)
        return _Posterior(spatial)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.public_decode_calls += 1
        raise AssertionError("Distilled STANDARD must use unclamped tiled decode")

    def decode_unclamped(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_lengths.append(int(latents.shape[2]))
        spatial = F.interpolate(latents[:, :1].float(), scale_factor=(1, 16, 16), mode="nearest")
        temporal = (
            spatial
            if spatial.shape[2] == 1
            else torch.cat((spatial[:, :, :1], spatial[:, :, 1:].repeat_interleave(4, dim=2)), dim=2)
        )
        return torch.cat((temporal, temporal * 0.5, -temporal), dim=1)


class _FakeCausalTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.in_channels = 2
        self.causal = True
        self.blocks = nn.ModuleList([nn.Identity()])
        self.calls: list[dict] = []
        self.to_calls: list[object] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)

    def init_causal_kv_caches(self):
        return [{"k": None, "v": None, "positions": [], "frames": [], "chunk_ids": []}]

    def forward(self, hidden_states, context, timestep, **kwargs):
        frames = list(kwargs["current_frames"])
        positions = list(kwargs["current_positions"])
        self.calls.append({
            "batch_size": int(context.shape[0]),
            "frames": frames,
            "positions": positions,
            "context_marker": float(context[-1, 0, 0]),
            "cache_frames": list(kwargs.get("cache_frames") or []),
            "mosaic_count": 0 if kwargs.get("mosaic_latents") is None else int(kwargs["mosaic_latents"].shape[2]),
            "write_cache": bool(kwargs.get("write_cache")),
            "timestep": timestep.detach().cpu(),
        })
        output = torch.zeros(
            (context.shape[0], *hidden_states.shape[1:]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if kwargs.get("write_cache"):
            chunk_ids = kwargs.get("current_cache_chunk_ids") or [-1] * len(frames)
            for cache in kwargs["kv_caches"]:
                new = torch.zeros(context.shape[0], len(frames), 1, dtype=hidden_states.dtype)
                cache["k"] = new if cache["k"] is None else torch.cat((cache["k"], new), dim=1)
                cache["v"] = new if cache["v"] is None else torch.cat((cache["v"], new), dim=1)
                cache["positions"].extend(positions)
                cache["frames"].extend(frames)
                cache["chunk_ids"].extend(chunk_ids)
        return output


class _FakeMemory:

    def __init__(self) -> None:
        self.append_sizes: list[int] = []
        self.query_count = 0

    def append(self, *, latents, w2c, intrinsics, **_kwargs) -> None:
        assert latents.shape[1] == len(w2c) == len(intrinsics)
        self.append_sizes.append(int(latents.shape[1]))

    def query(self, *, query_w2c, **_kwargs) -> MatrixGame35DistilledMemoryResult:
        self.query_count += 1
        return MatrixGame35DistilledMemoryResult(
            latents=torch.zeros(2, 3, 2, 2),
            valid_mask=torch.zeros(3, 2, 2, dtype=torch.bool),
            candidate_frame_ids=((0, 0, 0, 0, 0),) * 3,
            aligned_query_w2c=np.asarray(query_w2c, dtype=np.float32),
        )


class _FailFirstGeneratedAppendMemory(_FakeMemory):

    def __init__(self) -> None:
        super().__init__()
        self.append_attempts = 0
        self.query_anchors: list[np.ndarray] = []

    def append(self, *, latents, w2c, intrinsics, **kwargs) -> None:
        self.append_attempts += 1
        if self.append_attempts == 2:
            raise RuntimeError("Insufficient non-sky pixels for alignment")
        super().append(latents=latents, w2c=w2c, intrinsics=intrinsics, **kwargs)

    def query(self, *, anchor_w2c, **kwargs) -> MatrixGame35DistilledMemoryResult:
        self.query_anchors.append(np.asarray(anchor_w2c).copy())
        return super().query(**kwargs)


def _camera_file(tmp_path: Path, frame_count: int) -> str:
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    c2w[:, 0, 3] = np.linspace(0.0, 1.0, frame_count)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
    path = tmp_path / "camera.npz"
    np.savez(path, extrinsics_c2w=c2w, intrinsics=intrinsics)
    return str(path)


def _args(config):
    return SimpleNamespace(
        pipeline_config=config,
        disable_autocast=True,
        sp_size=1,
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
    )


@pytest.fixture(autouse=True)
def _force_cpu_runtime(monkeypatch):
    monkeypatch.setattr(distilled_standard_stages, "get_local_torch_device", lambda: torch.device("cpu"))


def test_distilled_vae_helpers_route_only_released_online_paths_to_tiling(monkeypatch) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig()
    operations = []

    def _capture_operation(_vae, value, *, operation, **_kwargs):
        operations.append(operation)
        return value

    monkeypatch.setattr(distilled_standard_stages, "run_matrixgame35_vae_operation", _capture_operation)
    stage = MatrixGame35DistilledRolloutStage(
        _FakeCausalTransformer(),
        _FakeVAE(),
        None,
        memory_factory=_FakeMemory,
    )
    args = _args(config)
    stage._encode_video(torch.zeros(1, 3, 1, 32, 32), args)
    stage._decode_video(torch.zeros(1, 2, 1, 2, 2), args)
    stage._memory_latents(np.zeros((1, 32, 32, 3), dtype=np.uint8), args)

    assert operations == [
        distilled_standard_stages.encode_matrixgame35_video,
        distilled_standard_stages.decode_matrixgame35_tiled_video,
        distilled_standard_stages.matrixgame35_tiled_memory_latents,
    ]
    assert config.vae_tiling is True


def test_distilled_dynamic_context_tiled_encodes_each_five_frame_window(monkeypatch) -> None:
    tiled_shapes = []

    def _tiled_encode(_vae, video):
        tiled_shapes.append(tuple(video.shape))
        return torch.zeros(1, 2, 2, 2, 2)

    def _execute_operation(vae, value, *, operation, **_kwargs):
        return operation(vae, value)

    monkeypatch.setattr(distilled_standard_stages, "encode_matrixgame35_tiled_video", _tiled_encode)
    monkeypatch.setattr(distilled_standard_stages, "run_matrixgame35_vae_operation", _execute_operation)
    stage = MatrixGame35DistilledRolloutStage(
        _FakeCausalTransformer(),
        _FakeVAE(),
        None,
        memory_factory=_FakeMemory,
    )
    entries = stage._dynamic_context_entries(
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((12, 32, 32, 3), dtype=np.uint8),
        chunk_index=0,
        target_w2c=np.repeat(np.eye(4, dtype=np.float32)[None], 12, axis=0),
        fastvideo_args=_args(MatrixGame35DistilledFirstPersonPipelineConfig()),
    )

    assert tiled_shapes == [(1, 3, 5, 32, 32)] * 3
    assert len(entries) == 3


def test_composition_and_full_fake_rollout_preserve_released_contract(tmp_path: Path) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _FakeCausalTransformer()
    vae = _FakeVAE()
    memory = _FakeMemory()
    stage = MatrixGame35DistilledRolloutStage(
        transformer,
        vae,
        None,
        memory_factory=lambda: memory,
    )
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 32, 32),
        camera_trajectory=_camera_file(tmp_path, 85),
        camera_convention="c2w",
        prompt_embeds=[torch.ones(1, 4, 8)],
        negative_prompt_embeds=[torch.zeros(1, 4, 8)],
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=17,
    )

    output = stage.forward(batch, _args(config))
    stats = output.extra["matrixgame35_distilled_stats"]

    assert output.output.device.type == "cpu"
    assert output.output.shape == (1, 3, 85, 32, 32)
    assert output.latents.device.type == "cpu"
    assert output.latents.shape == (1, 2, 22, 2, 2)
    assert stats["initial_noise_seeds"] == list(range(17, 24))
    assert stats["renoise_seeds"] == list(range(50017, 50024))
    assert stats["cache_frame_counts"] == [4, 7, 10, 13, 16, 19, 19]
    assert stats["context_positions"][0] is None
    assert all(position is not None for position in stats["context_positions"][1:])
    assert stats["memory_published_chunks"] == list(range(6))
    assert memory.append_sizes == [1, 12, 12, 12, 12, 12, 12]
    assert memory.query_count == 7
    assert stats["registration_decoded_chunks"] == list(range(6))
    assert vae.decode_lengths == [1] + [4] * 6 + [22]
    assert vae.public_decode_calls == 0
    assert vae.encode_shapes.count((1, 3, 1, 32, 32)) == 74
    assert vae.encode_shapes.count((1, 3, 5, 32, 32)) == 18
    assert len(transformer.calls) == 35  # bootstrap + 7 * (3 denoise + cache-fill) + 6 contexts
    assert transformer.calls[0]["frames"] == [0]
    assert transformer.calls[0]["positions"] == [1]
    assert config.vae_precision == "fp32"
    assert config.vae_tiling is True

    pipe = object.__new__(MatrixGame35DistilledFirstPersonPipeline)
    pipe.modules = {
        "text_encoder": object(),
        "tokenizer": object(),
        "vae": vae,
        "transformer": transformer,
    }
    pipe._stages = []
    pipe._stage_name_mapping = {}
    pipe._depth_adapter = None
    pipe.create_pipeline_stages(_args(config))
    assert list(pipe._stage_name_mapping) == [
        "input_validation_stage",
        "prompt_encoding_stage",
        "conditioning_stage",
        "distilled_rollout_stage",
    ]
    assert isinstance(pipe._stage_name_mapping["prompt_encoding_stage"], MatrixGame35TextEncodingStage)


def _run_profile_rollout(
    tmp_path: Path,
    *,
    profile: str,
    guidance_scale: float,
    hiar_scales: tuple[float, ...] = (),
):
    config = MatrixGame35DistilledFirstPersonPipelineConfig(
        matrixgame35_distilled_profile=profile,
        matrixgame35_distilled_hiar_scales=hiar_scales,
    )
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _FakeCausalTransformer()
    vae = _FakeVAE()
    memory = _FakeMemory()
    stage = MatrixGame35DistilledRolloutStage(
        transformer,
        vae,
        None,
        memory_factory=lambda: memory,
    )
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 32, 32),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt_embeds=[torch.ones(1, 4, 8)],
        negative_prompt_embeds=[torch.zeros(1, 4, 8)] if guidance_scale > 1.0 else [],
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=guidance_scale,
        seed=17,
    )
    return stage.forward(batch, _args(config)), transformer, vae, memory


def test_sink_anchor_profile_keeps_chunk_zero_context_free_then_reuses_c0(tmp_path: Path) -> None:
    output, transformer, vae, memory = _run_profile_rollout(
        tmp_path,
        profile="sink-anchor-context",
        guidance_scale=3.0,
    )
    stats = output.extra["matrixgame35_distilled_stats"]

    assert stats["profile"] == "sink-anchor-context"
    assert stats["context_positions"] == [None, 1, 1, 1, 1, 1, 1]
    assert stats["context_sources"] == [None] + ["forced_original_anchor"] * 6
    assert stats["hiar_prefix_noise"]["mode"] == "none"
    assert memory.append_sizes == [1, 12, 12, 12, 12, 12, 12]
    assert vae.encode_shapes.count((1, 3, 5, 32, 32)) == 0
    context_seed_calls = [
        call for call in transformer.calls
        if call["write_cache"] and call["frames"] == [0] and call["positions"] == [1]
    ]
    assert len(context_seed_calls) == 7  # persistent C0 plus one chunk-local C0 for chunks 1..6


def test_hiar_profile_rebuilds_noised_prefix_per_step_without_cfg(tmp_path: Path) -> None:
    output, transformer, _vae, _memory = _run_profile_rollout(
        tmp_path,
        profile="hiar-sde",
        guidance_scale=1.0,
        hiar_scales=(1.0, 0.5, 0.0),
    )
    stats = output.extra["matrixgame35_distilled_stats"]
    hiar = stats["hiar_prefix_noise"]

    assert stats["profile"] == "hiar-sde"
    assert stats["cache_frame_counts"] == [4, 7, 10, 13, 16, 19, 19]
    assert hiar["mode"] == "hiar_sde"
    assert hiar["dynamic_context_noised"] is True
    assert hiar["noise_scales_by_step"] == [1.0, 0.5, 0.0]
    assert len(hiar["chunks"]) == 7
    assert hiar["chunks"][0]["rolling_noise_seeds"] == [7_000_017, 7_000_118, 7_000_219]
    assert hiar["chunks"][0]["dynamic_context_noise_seeds"] == []
    assert hiar["chunks"][1]["rolling_noise_seeds"] == [7_010_017, 7_010_118, 7_010_219]
    assert hiar["chunks"][1]["dynamic_context_noise_seeds"] == [8_010_017, 8_010_118, 8_010_219]
    assert all(call["batch_size"] == 1 for call in transformer.calls)
    denoise_calls = [call for call in transformer.calls if not call["write_cache"]]
    assert len(denoise_calls) == 21


def test_input_validation_sets_released_negative_prompt_and_keeps_seed_override(tmp_path: Path) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig()
    batch = ForwardBatch(
        data_type="video",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 85),
        camera_convention="w2c",
        prompt="move forward",
        negative_prompt="",
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=99,
    )

    output = MatrixGame35DistilledInputValidationStage().forward(batch, _args(config))

    assert output.negative_prompt == DISTILLED_NEGATIVE_PROMPT
    assert output.seed == 99
    assert output.camera_convention == "w2c"
    assert output.pil_image.shape == (1, 3, 1, 704, 1280)
    assert output.section_prompts == ["move forward"]


def test_hiar_input_validation_requires_no_cfg_and_materializes_profile_policy(tmp_path: Path) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig(
        matrixgame35_distilled_profile="hiar-sde",
        matrixgame35_distilled_hiar_scales=(1.0, 0.5, 0.0),
    )
    batch = ForwardBatch(
        data_type="video",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt="move forward",
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=1.0,
        seed=3407,
    )

    output = MatrixGame35DistilledInputValidationStage().forward(batch, _args(config))

    assert output.do_classifier_free_guidance is False
    assert output.extra["matrixgame35_profile"] == "hiar-sde"
    assert output.extra["matrixgame35_hiar_scales"] == (1.0, 0.5, 0.0)

    bad_batch = ForwardBatch(
        data_type="video",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt="move forward",
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=3407,
    )
    with pytest.raises(ValueError, match="guidance_scale=1.0"):
        MatrixGame35DistilledInputValidationStage().forward(bad_batch, _args(config))


def test_distilled_validation_materializes_official_caption_json(tmp_path: Path) -> None:
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps({
            "detailed": {
                "0": {
                    "dynamic": "forward"
                }
            }
        }),
        encoding="utf-8",
    )
    batch = ForwardBatch(
        data_type="video",
        prompt=None,
        caption_path=str(caption_path),
        negative_prompt="",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 85),
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=3407,
    )

    output = MatrixGame35DistilledInputValidationStage().forward(
        batch,
        _args(MatrixGame35DistilledFirstPersonPipelineConfig()),
    )

    assert output.prompt is None
    assert output.caption_path == str(caption_path)
    assert output.section_prompts == ["forward"]


def test_two_section_rollout_switches_context_every_seven_chunks(tmp_path: Path) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _FakeCausalTransformer()
    stage = MatrixGame35DistilledRolloutStage(
        transformer,
        _FakeVAE(),
        None,
        memory_factory=_FakeMemory,
    )
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 32, 32),
        camera_trajectory=_camera_file(tmp_path, 169),
        prompt_embeds=[torch.stack((torch.ones(4, 8), torch.full((4, 8), 2.0)))],
        negative_prompt_embeds=[torch.zeros(1, 4, 8)],
        height=704,
        width=1280,
        num_frames=169,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=17,
    )

    stage.forward(batch, _args(config))

    generation_markers = [call["context_marker"] for call in transformer.calls if len(call["frames"]) == 3]
    assert generation_markers == [1.0] * (7 * 4) + [2.0] * (7 * 4)


def test_failed_generated_registration_keeps_last_published_memory_anchor(tmp_path: Path) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    memory = _FailFirstGeneratedAppendMemory()
    stage = MatrixGame35DistilledRolloutStage(
        _FakeCausalTransformer(),
        _FakeVAE(),
        None,
        memory_factory=lambda: memory,
    )
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 32, 32),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt_embeds=[torch.ones(1, 4, 8)],
        negative_prompt_embeds=[torch.zeros(1, 4, 8)],
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=17,
    )

    output = stage.forward(batch, _args(config))

    assert len(memory.query_anchors) == 7
    np.testing.assert_array_equal(memory.query_anchors[0], memory.query_anchors[1])
    assert output.extra["matrixgame35_distilled_stats"]["memory_published_chunks"] == list(range(1, 6))


def test_distilled_rollout_honors_module_cpu_offload_boundaries(tmp_path: Path) -> None:
    config = MatrixGame35DistilledFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _FakeCausalTransformer()
    vae = _FakeVAE()
    fastvideo_args = _args(config)
    fastvideo_args.vae_cpu_offload = True
    fastvideo_args.dit_cpu_offload = True
    stage = MatrixGame35DistilledRolloutStage(transformer, vae, None, memory_factory=_FakeMemory)
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 32, 32),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt_embeds=[torch.ones(1, 4, 8)],
        negative_prompt_embeds=[torch.zeros(1, 4, 8)],
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=3,
        guidance_scale=3.0,
        seed=17,
    )

    stage.forward(batch, fastvideo_args)

    assert transformer.to_calls == [torch.device("cpu"), "cpu"] * 8
    assert vae.to_calls[::2] == [torch.device("cpu")] * 22
    assert vae.to_calls[1::2] == ["cpu"] * 22
