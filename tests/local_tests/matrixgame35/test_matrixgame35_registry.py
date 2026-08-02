# SPDX-License-Identifier: Apache-2.0
"""Public registry coverage for the activated Matrix-Game 3.5 variants."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torchvision")

from fastvideo.api.presets import get_preset
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.matrixgame35 import (
    MatrixGame35BaseFirstPersonPipelineConfig,
    MatrixGame35BaseThirdPersonPipelineConfig,
    MatrixGame35DistilledFirstPersonPipelineConfig,
)
from fastvideo.pipelines.basic.matrixgame35.base_first_person_pipeline import (
    MatrixGame35BaseFirstPersonPipeline,
)
from fastvideo.pipelines.basic.matrixgame35.base_third_person_pipeline import (
    MatrixGame35BaseThirdPersonPipeline,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_pipeline import (
    MatrixGame35DistilledFirstPersonPipeline,
)
from fastvideo.registry import get_default_preset, get_model_info, get_pipeline_config_cls_from_name

_VARIANTS = (
    (
        "FastVideo/Matrix-Game-3.5-Base-First-Person-Diffusers",
        "MatrixGame35BaseFirstPersonPipeline",
        MatrixGame35BaseFirstPersonPipelineConfig,
        MatrixGame35BaseFirstPersonPipeline,
        "matrixgame35_base_first_person",
        25,
        5.0,
    ),
    (
        "FastVideo/Matrix-Game-3.5-Base-Third-Person-Diffusers",
        "MatrixGame35BaseThirdPersonPipeline",
        MatrixGame35BaseThirdPersonPipelineConfig,
        MatrixGame35BaseThirdPersonPipeline,
        "matrixgame35_base_third_person",
        25,
        5.0,
    ),
    (
        "FastVideo/Matrix-Game-3.5-Distilled-First-Person-Diffusers",
        "MatrixGame35DistilledFirstPersonPipeline",
        MatrixGame35DistilledFirstPersonPipelineConfig,
        MatrixGame35DistilledFirstPersonPipeline,
        "matrixgame35_distilled_first_person",
        3,
        3.0,
    ),
)


@pytest.mark.parametrize(("model_id", "_pipeline_name", "config_cls", "_pipeline_cls", "preset_name", "steps", "cfg"),
                         _VARIANTS)
def test_matrixgame35_exact_ids_resolve_public_defaults(
    model_id: str,
    _pipeline_name: str,
    config_cls: type,
    _pipeline_cls: type,
    preset_name: str,
    steps: int,
    cfg: float,
) -> None:
    assert get_pipeline_config_cls_from_name(model_id) is config_cls
    assert get_default_preset(model_id) == preset_name
    sampling = SamplingParam.from_pretrained(model_id)
    assert (sampling.height, sampling.width, sampling.num_frames, sampling.fps) == (704, 1280, 85, 16)
    assert (sampling.num_inference_steps, sampling.guidance_scale, sampling.seed) == (steps, cfg, 3407)


@pytest.mark.parametrize(("_model_id", "pipeline_name", "config_cls", "pipeline_cls", "preset_name", "_steps", "_cfg"),
                         _VARIANTS)
def test_matrixgame35_converted_model_index_resolves_pipeline(
    tmp_path: Path,
    _model_id: str,
    pipeline_name: str,
    config_cls: type,
    pipeline_cls: type,
    preset_name: str,
    _steps: int,
    _cfg: float,
) -> None:
    model_dir = tmp_path / pipeline_name
    model_dir.mkdir()
    (model_dir / "transformer").mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": pipeline_name,
            "_diffusers_version": "0.35.0.dev0",
        }),
        encoding="utf-8",
    )

    info = get_model_info(str(model_dir))
    assert info.pipeline_config_cls is config_cls
    assert info.pipeline_cls is pipeline_cls
    assert get_default_preset(str(model_dir)) == preset_name


@pytest.mark.parametrize(("_model_id", "_pipeline_name", "_config_cls", "_pipeline_cls", "preset_name", "steps", "cfg"),
                         _VARIANTS)
def test_matrixgame35_presets_are_registered(
    _model_id: str,
    _pipeline_name: str,
    _config_cls: type,
    _pipeline_cls: type,
    preset_name: str,
    steps: int,
    cfg: float,
) -> None:
    preset = get_preset(preset_name, "matrixgame35")
    assert preset.workload_type == "i2v"
    assert preset.defaults["num_inference_steps"] == steps
    assert preset.defaults["guidance_scale"] == cfg
