# SPDX-License-Identifier: Apache-2.0
"""Direct tiny parity for the distilled Matrix-Game 3.5 causal cache path."""

from pathlib import Path

import pytest
import torch

import fastvideo.attention.layer as attention_layer
from fastvideo.attention.backends.sdpa import SDPABackend
from fastvideo.configs.models.dits.matrixgame35 import (
    MatrixGame35WanVideoArchConfig,
    MatrixGame35WanVideoConfig,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.matrixgame35 import MatrixGame35Transformer3DModel
from fastvideo.models.loader.utils import get_param_names_mapping
from tests.local_tests.matrixgame35._upstream import (
    load_upstream_pipeline,
    load_upstream_transformer,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"


def _camera_info(frame_count: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    projection = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, frame_count, 4, 1, 1)
    for frame in range(frame_count):
        for subframe in range(4):
            projection[0, frame, subframe, 0, 0] = 1.0 + 0.01 * frame
            projection[0, frame, subframe, 1, 1] = 1.0 + 0.01 * subframe
            projection[0, frame, subframe, 0, 3] = 0.02 * (frame + subframe)
    projection_transpose = projection.transpose(-1, -2).contiguous()
    projection_inverse = torch.linalg.inv(projection)
    return projection, (projection, projection_transpose, projection_inverse)


def _native_model(official: torch.nn.Module) -> MatrixGame35Transformer3DModel:
    arch = MatrixGame35WanVideoArchConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=128,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=256,
        num_layers=1,
        subject_ref_memory_max_refs=0,
        causal=True,
    )
    model = MatrixGame35Transformer3DModel(
        MatrixGame35WanVideoConfig(arch_config=arch),
        hf_config={},
    )
    map_name = get_param_names_mapping(model.param_names_mapping)
    mapped_state = {map_name(name)[0]: value for name, value in official.state_dict().items()}
    incompatible = model.load_state_dict(mapped_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    return model.eval()


def _clone_caches(caches: list[dict]) -> list[dict]:
    return [
        {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else list(value)
            for name, value in cache.items()
        }
        for cache in caches
    ]


def _assert_caches_close(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for actual_cache, expected_cache in zip(actual, expected):
        assert set(actual_cache) == set(expected_cache)
        for name in ("positions", "frames", "chunk_ids"):
            assert actual_cache[name] == expected_cache[name]
        for name in ("k", "v"):
            if expected_cache[name] is None:
                assert actual_cache[name] is None
            else:
                torch.testing.assert_close(actual_cache[name], expected_cache[name], rtol=2e-5, atol=2e-5)


def test_causal_cache_lifecycle_and_mosaic_holes_match_pinned_model_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _OFFICIAL_DIR.is_dir():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {_OFFICIAL_DIR}")
    upstream_model = load_upstream_transformer(_OFFICIAL_DIR).wan_video_dit
    upstream_pipeline = load_upstream_pipeline(_OFFICIAL_DIR)

    torch.manual_seed(3510)
    official = upstream_model.WanModel(
        dim=128,
        in_dim=4,
        ffn_dim=256,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=1,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        use_prope=True,
        subject_ref_memory_enabled=False,
    ).eval()
    monkeypatch.setattr(attention_layer, "get_attn_backend", lambda *args, **kwargs: SDPABackend)
    native = _native_model(official)

    generator = torch.Generator().manual_seed(3511)
    anchor = torch.randn(1, 4, 1, 4, 4, generator=generator)
    noisy = torch.randn(1, 4, 2, 4, 4, generator=generator)
    clean = torch.randn(1, 4, 2, 4, 4, generator=generator)
    later = torch.randn(1, 4, 2, 4, 4, generator=generator)
    mosaic = torch.randn(1, 4, 2, 4, 4, generator=generator)
    mosaic[:, :, 0, :2, :2] = 0
    context = torch.randn(1, 3, 16, generator=generator)
    camera_info = _camera_info(frame_count=7)
    official_caches = upstream_pipeline.init_causal_kv_caches(1)
    native_caches = native.init_causal_kv_caches()
    assert native_caches == [{"k": None, "v": None, "positions": [], "frames": [], "chunk_ids": []}]

    def compare_forward(
        latents: torch.Tensor,
        timestep_frames: torch.Tensor,
        *,
        current_positions: list[int],
        current_frames: list[int],
        mosaic_latents: torch.Tensor | None = None,
        mosaic_positions: list[int] | None = None,
        mosaic_frames: list[int] | None = None,
        cache_positions: list[int] | None = None,
        cache_frames: list[int] | None = None,
        cache_read_chunk_id: int | None = None,
        current_cache_chunk_ids: list[int] | None = None,
        write_cache: bool = False,
    ) -> None:
        with torch.inference_mode():
            expected = upstream_pipeline.model_fn_causal_kv(
                official,
                latents_chunk=latents,
                timestep_frames=timestep_frames,
                context=context,
                camera_info=camera_info,
                cur_positions=current_positions,
                cur_frames=current_frames,
                caches=official_caches,
                mosaic_latent=mosaic_latents,
                mosaic_positions=mosaic_positions,
                mosaic_frames=mosaic_frames,
                cache_positions=cache_positions,
                cache_frames=cache_frames,
                cache_read_chunk_id=cache_read_chunk_id,
                cur_cache_chunk_ids=current_cache_chunk_ids,
                write_cache=write_cache,
            )
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            actual = native(
                hidden_states=latents,
                encoder_hidden_states=context,
                timestep=timestep_frames,
                camera_info=camera_info,
                kv_caches=native_caches,
                mosaic_latents=mosaic_latents,
                current_positions=current_positions,
                current_frames=current_frames,
                mosaic_positions=mosaic_positions,
                mosaic_frames=mosaic_frames,
                cache_positions=cache_positions,
                cache_frames=cache_frames,
                cache_read_chunk_id=cache_read_chunk_id,
                current_cache_chunk_ids=current_cache_chunk_ids,
                write_cache=write_cache,
            )
        max_abs_diff = float((actual - expected).abs().max().item())
        print(f"Matrix-Game 3.5 causal model_fn max_abs_diff={max_abs_diff:.8g}")
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        _assert_caches_close(native_caches, official_caches)

    compare_forward(
        anchor,
        torch.zeros(1),
        current_positions=[0],
        current_frames=[0],
        current_cache_chunk_ids=[-1],
        write_cache=True,
    )
    assert native_caches[0]["k"] is not None
    assert native_caches[0]["k"].shape[1] == 4

    for timestep in (1000.0, 667.0):
        before_native = _clone_caches(native_caches)
        before_official = _clone_caches(official_caches)
        compare_forward(
            noisy,
            torch.tensor([0.0, 0.0, timestep, timestep]),
            current_positions=[3, 4],
            current_frames=[3, 4],
            mosaic_latents=mosaic,
            mosaic_positions=[1, 2],
            mosaic_frames=[1, 2],
            cache_read_chunk_id=0,
            current_cache_chunk_ids=[0, 0],
            write_cache=False,
        )
        _assert_caches_close(native_caches, before_native)
        _assert_caches_close(official_caches, before_official)

    compare_forward(
        clean,
        torch.zeros(4),
        current_positions=[3, 4],
        current_frames=[3, 4],
        mosaic_latents=mosaic,
        mosaic_positions=[1, 2],
        mosaic_frames=[1, 2],
        cache_read_chunk_id=0,
        current_cache_chunk_ids=[0, 0],
        write_cache=True,
    )
    assert native_caches[0]["positions"] == [0, 3, 4]
    assert native_caches[0]["frames"] == [0, 3, 4]
    assert native_caches[0]["chunk_ids"] == [-1, 0, 0]

    compare_forward(
        later,
        torch.tensor([333.0, 333.0]),
        current_positions=[3, 4],
        current_frames=[5, 6],
        cache_positions=[0, 1, 2],
        cache_frames=[0, 3, 4],
        cache_read_chunk_id=0,
        current_cache_chunk_ids=[1, 2],
        write_cache=False,
    )
    compare_forward(
        later,
        torch.tensor([333.0, 333.0]),
        current_positions=[3, 4],
        current_frames=[5, 6],
        cache_positions=[0, 1, 2],
        cache_frames=[0, 3, 4],
        cache_read_chunk_id=1,
        current_cache_chunk_ids=[1, 2],
        write_cache=False,
    )
