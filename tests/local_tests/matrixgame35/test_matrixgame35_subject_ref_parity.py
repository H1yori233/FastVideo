# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.configs.models.dits.matrixgame35 import MatrixGame35WanVideoArchConfig, MatrixGame35WanVideoConfig
from fastvideo.models.dits.matrixgame35 import MatrixGame35Transformer3DModel
from fastvideo.pipelines.basic.matrixgame35.conditioning import (
    build_mosaic_cross_attention_keep_mask,
    build_subject_ref_memory_tokens,
    prepend_subject_ref_prope_camera_info,
)
from tests.local_tests.matrixgame35._upstream import load_upstream_transformer


PARITY_SCOPE = "implementation_subcomponent"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"


class _ImportOnlyType:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _install_module(name: str, **attributes: Any) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = module
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _install_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


@lru_cache(maxsize=1)
def _load_upstream_pipeline() -> ModuleType:
    source_file = _OFFICIAL_DIR / "diffsynth" / "pipelines" / "wan_video.py"
    if not source_file.is_file():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {source_file}")

    transformer_modules = load_upstream_transformer(_OFFICIAL_DIR)
    models_name = transformer_modules.wan_video_dit.__name__.rsplit(".", 1)[0]
    diffsynth_name = models_name.rsplit(".", 1)[0]

    core = sys.modules[f"{diffsynth_name}.core"]
    core.ModelConfig = _ImportOnlyType
    core.gradient_checkpoint_forward = lambda *_args, **_kwargs: None
    _install_namespace(f"{diffsynth_name}.core.device", _OFFICIAL_DIR / "diffsynth" / "core" / "device")
    _install_module(f"{diffsynth_name}.core.device.npu_compatible_device", get_device_type=lambda: "cpu")
    _install_module(f"{diffsynth_name}.diffusion", FlowMatchScheduler=_ImportOnlyType)
    _install_module(
        f"{diffsynth_name}.diffusion.base_pipeline",
        BasePipeline=_ImportOnlyType,
        PipelineUnit=_ImportOnlyType,
    )

    import_only_models = {
        "wan_video_dit_s2v": {"rope_precompute": lambda *_args, **_kwargs: None},
        "wan_video_text_encoder": {
            "WanTextEncoder": _ImportOnlyType,
            "HuggingfaceTokenizer": _ImportOnlyType,
        },
        "wan_video_vae": {"WanVideoVAE": _ImportOnlyType},
        "wan_video_image_encoder": {"WanImageEncoder": _ImportOnlyType},
        "wan_video_vace": {"VaceWanModel": _ImportOnlyType},
        "wan_video_motion_controller": {"WanMotionControllerModel": _ImportOnlyType},
        "wan_video_animate_adapter": {"WanAnimateAdapter": _ImportOnlyType},
        "wan_video_mot": {"MotWanModel": _ImportOnlyType},
        "wav2vec": {"WanS2VAudioEncoder": _ImportOnlyType},
        "longcat_video_dit": {"LongCatVideoTransformer3DModel": _ImportOnlyType},
    }
    for module_name, attributes in import_only_models.items():
        _install_module(f"{models_name}.{module_name}", **attributes)

    _install_namespace(f"{diffsynth_name}.pipelines", _OFFICIAL_DIR / "diffsynth" / "pipelines")
    module_name = f"{diffsynth_name}.pipelines.wan_video"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned upstream pipeline source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source_file.resolve():
        raise RuntimeError(f"Loaded unexpected upstream pipeline source: {module.__file__}")
    return module


def _complex_frequencies(rows: int, width: int, scale: float) -> torch.Tensor:
    positions = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    bands = torch.arange(1, width + 1, dtype=torch.float32).unsqueeze(0)
    phase = positions * bands * scale
    return torch.polar(torch.ones_like(phase), phase)


def _make_dit(*, max_refs: int, local_size: int, expose_patchify: bool) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(3500 + max_refs + local_size)
    patch_embedding = torch.nn.Conv3d(2, 6, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    with torch.no_grad():
        patch_embedding.weight.copy_(torch.randn(patch_embedding.weight.shape, generator=generator))
        patch_embedding.bias.copy_(torch.randn(patch_embedding.bias.shape, generator=generator))
    dit = SimpleNamespace(
        subject_ref_memory_enabled=True,
        subject_ref_index_embedding=torch.randn(max_refs, 6, generator=generator),
        subject_ref_type_embedding=torch.randn(1, 6, generator=generator),
        subject_ref_local_h_embedding=torch.randn(local_size, 6, generator=generator),
        subject_ref_local_w_embedding=torch.randn(local_size, 6, generator=generator),
        patch_embedding=patch_embedding,
        freqs=(
            _complex_frequencies(32, 2, 0.13),
            _complex_frequencies(32, 2, 0.17),
            _complex_frequencies(32, 2, 0.19),
        ),
    )
    if expose_patchify:
        dit.patchify = patch_embedding
    return dit


def _without_patchify(dit: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(**{name: value for name, value in vars(dit).items() if name != "patchify"})


def _assert_memory_equal(actual: dict[str, Any] | None, expected: dict[str, Any] | None) -> None:
    assert actual is not None
    assert expected is not None
    assert actual.keys() == expected.keys()
    for key in ("token_count", "ref_count", "slot_grid", "slot_start"):
        assert actual[key] == expected[key]
    torch.testing.assert_close(actual["x"], expected["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual["freqs"], expected["freqs"], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", ("rchw", "rc1hw", "bcrhw", "batched_bcrhw"))
@pytest.mark.parametrize("max_refs", (2, 4))
def test_subject_ref_memory_matches_upstream_for_released_layouts_and_slot_limits(
    layout: str,
    max_refs: int,
) -> None:
    upstream = _load_upstream_pipeline()
    official_dit = _make_dit(max_refs=max_refs, local_size=3, expose_patchify=True)
    fastvideo_dit = _without_patchify(official_dit)
    references = torch.randn(5, 2, 8, 12, generator=torch.Generator().manual_seed(35))
    if layout == "rc1hw":
        references = references.unsqueeze(2)
    elif layout == "bcrhw":
        references = references.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    elif layout == "batched_bcrhw":
        references = references.permute(1, 0, 2, 3).unsqueeze(0).repeat(2, 1, 1, 1, 1).contiguous()
        references[1].add_(0.25)

    kwargs = dict(
        batch_size=2,
        video_h=8,
        video_w=12,
        subject_ref_slot_ratio=0.5,
        subject_ref_time_gap=3,
        device="cpu",
        dtype=torch.float32,
    )
    expected = upstream._build_subject_ref_memory_tokens(official_dit, references, **kwargs)
    actual = build_subject_ref_memory_tokens(fastvideo_dit, references, **kwargs)

    _assert_memory_equal(actual, expected)
    assert actual is not None
    assert actual["ref_count"] == max_refs
    assert actual["slot_grid"] == (2, 2)
    assert actual["slot_start"] == (2, 4)
    assert actual["x"].shape == (2, max_refs * 4, 6)


def test_subject_ref_local_interpolation_and_negative_time_rope_match_upstream() -> None:
    upstream = _load_upstream_pipeline()
    dit = _make_dit(max_refs=2, local_size=2, expose_patchify=True)
    references = torch.randn(2, 2, 10, 14, generator=torch.Generator().manual_seed(36))
    kwargs = dict(
        batch_size=1,
        video_h=10,
        video_w=14,
        subject_ref_slot_ratio=1.0,
        subject_ref_time_gap=2,
        device="cpu",
        dtype=torch.float32,
    )

    expected = upstream._build_subject_ref_memory_tokens(dit, references, **kwargs)
    actual = build_subject_ref_memory_tokens(dit, references, **kwargs)

    _assert_memory_equal(actual, expected)
    assert actual is not None
    assert actual["slot_grid"] == (5, 5)
    frequencies = actual["freqs"].reshape(2, 5, 5, 1, -1)
    expected_negative_time = dit.freqs[0][torch.tensor([2, 4])].conj()
    torch.testing.assert_close(
        frequencies[..., :2],
        expected_negative_time.view(2, 1, 1, 1, 2).expand(2, 5, 5, 1, 2),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.any(frequencies[..., :2].imag < 0)


def test_production_model_reports_missing_arbitrary_position_rope_carrier() -> None:
    arch_config = MatrixGame35WanVideoArchConfig(
        num_attention_heads=1,
        attention_head_dim=64,
        in_channels=2,
        out_channels=2,
        text_dim=8,
        freq_dim=16,
        ffn_dim=128,
        num_layers=0,
        subject_ref_memory_max_refs=2,
    )
    model = MatrixGame35Transformer3DModel(MatrixGame35WanVideoConfig(arch_config=arch_config), {})
    references = torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(38))

    assert not hasattr(model, "freqs")
    with pytest.raises(ValueError, match="three native RoPE frequency tables"):
        build_subject_ref_memory_tokens(
            model,
            references,
            batch_size=1,
            video_h=4,
            video_w=4,
            subject_ref_slot_ratio=0.5,
            subject_ref_time_gap=1,
            device="cpu",
            dtype=torch.float32,
        )


def _camera_info() -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    generator = torch.Generator().manual_seed(37)
    projection = torch.randn(1, 2, 4, 4, 4, generator=generator)
    projection_transpose = torch.randn(1, 2, 4, 4, 4, generator=generator)
    projection_inverse = torch.randn(1, 2, 4, 4, 4, generator=generator)
    w2c = torch.randn(1, 2, 4, 4, 4, generator=generator)
    view_change_positions = torch.randn(1, 6, 3, generator=generator)
    return w2c, (projection, projection_transpose, projection_inverse), view_change_positions


@pytest.mark.parametrize("mode,anchor_index", (("identity", None), ("clean_anchor", 4)))
def test_subject_ref_prope_prefix_cameras_match_upstream(mode: str, anchor_index: int | None) -> None:
    upstream = _load_upstream_pipeline()
    camera_info = _camera_info()
    kwargs = dict(
        prefix_token_count=5,
        tokens_per_frame=3,
        frame_count=2,
        mode=mode,
        clean_anchor_token_index=anchor_index,
    )

    expected = upstream._prepend_subject_ref_prope_camera_info(camera_info, **kwargs)
    actual = prepend_subject_ref_prope_camera_info(camera_info, **kwargs)

    assert actual is not None
    assert expected is not None
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    for actual_matrix, expected_matrix in zip(actual[1], expected[1]):
        torch.testing.assert_close(actual_matrix, expected_matrix, rtol=0.0, atol=0.0)
        assert actual_matrix.shape == (1, 11, 4, 4, 4)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
    if mode == "identity":
        identity = torch.eye(4).reshape(1, 1, 1, 4, 4).expand(1, 5, 4, 4, 4)
        torch.testing.assert_close(actual[1][0][:, :5], identity, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[2][:, :5, 0], torch.ones(1, 5), rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[2][:, :5, 1:], torch.zeros(1, 5, 2), rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(
            actual[1][0][:, :5],
            actual[1][0][:, 9:10].expand(1, 5, 4, 4, 4),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "counts",
    (
        (0, 3, 1, 0, 2, 4),
        (5, 0, 1, 2, 3, 4),
        (5, 3, 1, 2, 3, 4),
    ),
)
def test_subject_ref_cross_attention_keep_mask_matches_upstream(counts: tuple[int, ...]) -> None:
    upstream = _load_upstream_pipeline()
    prefix, reference, first, mosaic, noisy, tokens = counts
    kwargs = dict(
        prefix_memory_token_count=prefix,
        reference_token_count=reference,
        first_frame_count=first,
        mosaic_frame_count=mosaic,
        noisy_frame_count=noisy,
        tokens_per_frame=tokens,
        device="cpu",
    )
    expected = upstream._build_mosaic_cross_attn_keep_mask(**kwargs)
    actual = build_mosaic_cross_attention_keep_mask(**kwargs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    if prefix:
        assert not actual[:prefix].any()
