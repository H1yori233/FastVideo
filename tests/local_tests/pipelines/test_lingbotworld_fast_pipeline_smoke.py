# SPDX-License-Identifier: Apache-2.0
"""Smoke / preflight tests for the LingBot-World-Fast causal-DMD I2V pipeline.

The preflight validates import, registry, preset, and config wiring. The full
load/generate smoke loads the real pipeline through ``VideoGenerator`` and
generates a single-chunk clip; it activates when CUDA is available and the
``robbyant/lingbot-world-fast-diffusers`` weights are reachable (cached or
downloadable), or when ``$LINGBOT_FAST_DIR`` points at a local copy.

Usage:
    DISABLE_SP=1 pytest tests/local_tests/pipelines/test_lingbotworld_fast_pipeline_smoke.py -v -s
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

MODEL_ID = os.getenv("LINGBOT_FAST_DIR", "robbyant/lingbot-world-fast-diffusers")

requires_runtime = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LingBot-World-Fast pipeline imports require the CUDA/kernel runtime")


@requires_runtime
def test_lingbotworld_fast_preflight() -> None:
    """Import + registry + preset + config wiring (no heavy weights)."""
    import fastvideo  # noqa: F401
    from fastvideo.registry import get_model_family
    from fastvideo.configs.pipelines.lingbotworld import LingBotWorldFastI2V480PConfig
    from fastvideo.pipelines.basic.lingbotworld.lingbotworld_fast_pipeline import (
        EntryClass, LingBotWorldCausalDMDPipeline)
    from fastvideo.pipelines.stages import LingBotWorldFastCausalDenoisingStage  # noqa: F401

    assert get_model_family("robbyant/lingbot-world-fast-diffusers") == "lingbotworld_fast"
    # The Cam detector must NOT claim the Fast repo.
    assert get_model_family("FastVideo/LingBot-World-Base-Cam-Diffusers") == "lingbotworld"
    assert EntryClass is LingBotWorldCausalDMDPipeline

    cfg = LingBotWorldFastI2V480PConfig()
    assert cfg.is_causal is True
    assert cfg.boundary_ratio is None
    assert cfg.num_frames_per_block == 3
    assert cfg.dit_config.arch_config.in_channels == 36
    assert cfg.dit_config.arch_config.out_channels == 16
    assert cfg.vae_config.load_encoder is True


def _weights_available() -> bool:
    if os.path.isdir(MODEL_ID):
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID)
        return True
    except Exception:
        return False


@requires_runtime
def test_lingbotworld_fast_pipeline_smoke() -> None:
    """Load the real pipeline and generate a single block-causal chunk."""
    if not _weights_available():
        pytest.skip(f"weights for {MODEL_ID} unavailable")

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )

    result = generator.generate_video(
        "a calm river winding through a green valley under a bright sky",
        image_path="https://raw.githubusercontent.com/Robbyant/lingbot-world/main/examples/00/image.jpg",
        num_frames=9,          # -> 3 latent frames = 1 causal chunk
        height=480,
        width=832,
        save_video=False,
        return_frames=True,
    )

    if isinstance(result, list):
        result = result[0]
    frames = np.asarray(result["frames"])
    assert frames.size > 0, "empty output"
    assert np.isfinite(frames).all(), "non-finite pixels in output"
    assert frames.std() > 1.0, f"output looks degenerate (std={frames.std():.3f})"
    print(f"\n[smoke] frames shape={frames.shape} dtype={frames.dtype} "
          f"min={frames.min():.1f} max={frames.max():.1f} std={frames.std():.1f}")
