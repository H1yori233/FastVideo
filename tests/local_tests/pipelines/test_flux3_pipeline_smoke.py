# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 registry and real public-API smoke on one 32 GiB CUDA GPU."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
MODEL = Path(os.getenv("FLUX3_MODEL_DIR", ROOT / "official_weights/flux3/flux-3-image-klein"))


@pytest.fixture(scope="module")
def model_path():
    if not (MODEL / "modular_model_index.json").exists():
        pytest.skip("Official FLUX.3 weights are required")
    return str(MODEL)


def test_flux3_pipeline_registration(model_path):
    from fastvideo import registry
    from fastvideo.api.presets import get_preset
    from fastvideo.configs.pipelines.flux3 import Flux3PipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.flux3.flux3_pipeline import EntryClass, Flux3Pipeline

    assert EntryClass is Flux3Pipeline
    info = registry.get_model_info(model_path, workload_type=WorkloadType.T2I)
    assert info.pipeline_cls is Flux3Pipeline
    assert info.pipeline_config_cls is Flux3PipelineConfig
    preset_name, family = registry.get_preset_selection(model_path)
    assert family == "flux3"
    preset = get_preset(preset_name, family)
    assert preset.defaults["num_inference_steps"] == 16
    assert preset.defaults["num_frames"] == 1
    assert preset.defaults["guidance_scale"] == 1.0
    assert preset.defaults["flux3_upsample_prompt"] is True


def test_flux3_public_generate_with_prompt_upsampling(tmp_path, model_path):
    if not torch.cuda.is_available():
        pytest.skip("A 32 GiB CUDA GPU is required")
    from fastvideo import VideoGenerator
    from fastvideo.api import EngineConfig, GeneratorConfig, OffloadConfig, PipelineSelection
    from fastvideo.api.results import GenerationResult

    generator = VideoGenerator.from_config(GeneratorConfig(
        model_path=model_path,
        engine=EngineConfig(num_gpus=1, use_fsdp_inference=False, offload=OffloadConfig(
            dit=False, dit_layerwise=False, text_encoder=False, vae=False,
            pin_cpu_memory=False, lazy_module_load=True)),
        pipeline=PipelineSelection(workload_type="t2i"),
    ))
    try:
        result = generator.generate({
            "prompt": "A red fox resting beside a quiet lake.",
            "sampling": {"height": 256, "width": 256, "num_frames": 1, "num_inference_steps": 4,
                         "guidance_scale": 1.0, "seed": 123},
            "output": {"save_video": True, "return_frames": True, "output_path": str(tmp_path / "generated.png")},
            "extensions": {"flux3_upsample_prompt": True, "flux3_pup_max_new_tokens": 16,
                           "flux3_auto_size": False},
        })
        assert isinstance(result, GenerationResult)
        assert result.samples is not None
        samples = torch.as_tensor(result.samples)
        assert tuple(samples.shape[-2:]) == (256, 256)
        assert torch.isfinite(samples).all()
        assert result.frames is not None
        assert result.size == (256, 256, 1)
        assert np.asarray(result.frames).shape == (1, 256, 256, 3)
        assert result.video_path is not None and Path(result.video_path).suffix == ".png"
        with Image.open(result.video_path) as image:
            assert image.format == "PNG" and image.size == (256, 256)
            np.testing.assert_array_equal(np.asarray(image), result.frames[0])
        assert result.prompt
        cropped = generator.generate({
            "prompt": "A red balloon.",
            "inputs": {"references": [Image.new("RGB", (65, 97), "red")]},
            "sampling": {"num_inference_steps": 2},
            "extensions": {"flux3_auto_size": True, "flux3_upsample_prompt": False},
            "output": {"save_video": True, "return_frames": True, "output_path": str(tmp_path / "cropped.png")},
        })
        assert cropped.size == (97, 65, 1)
        assert cropped.frames is not None
        assert np.asarray(cropped.frames).shape == (1, 97, 65, 3)
        assert cropped.video_path is not None and Path(cropped.video_path).suffix == ".png"
        with Image.open(cropped.video_path) as image:
            assert image.format == "PNG" and image.size == (65, 97)
            np.testing.assert_array_equal(np.asarray(image), cropped.frames[0])
    finally:
        result = cropped = samples = None
        generator.shutdown()
