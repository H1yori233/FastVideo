# SPDX-License-Identifier: Apache-2.0
"""Smoke / preflight tests for the LingBot-World-Act full-sequence A14B-MoE I2V pipeline.

The preflight validates import, registry, preset, and config wiring (act2cam =
control_dim 7 on the Cam MoE architecture). The full load/generate smoke loads
the real pipeline through ``VideoGenerator``, builds the 7-channel act2cam
control from an action string, and generates a short clip; it activates when CUDA
is available and ``$LINGBOT_ACT_DIR`` points at a converted Act diffusers copy
(produced by ``scripts/checkpoint_conversion/convert_lingbotworld_act_to_diffusers.py``).

Usage:
    DISABLE_SP=1 LINGBOT_ACT_DIR=official_weights/lingbotworld_act_diffusers \
        pytest tests/local_tests/pipelines/test_lingbotworld_act_pipeline_smoke.py -v -s
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

MODEL_ID = os.getenv("LINGBOT_ACT_DIR", "FastVideo/LingBot-World-Base-Act-Diffusers")

requires_runtime = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LingBot-World-Act pipeline imports require the CUDA/kernel runtime")


@requires_runtime
def test_lingbotworld_act_preflight() -> None:
    """Import + registry + preset + config wiring (no heavy weights)."""
    import fastvideo  # noqa: F401
    from fastvideo.registry import get_model_family
    from fastvideo.configs.pipelines.lingbotworld import LingBotWorldActI2V480PConfig

    assert get_model_family("FastVideo/LingBot-World-Base-Act-Diffusers") == "lingbotworld_act"
    # The Cam and Fast detectors must NOT claim the Act repo.
    assert get_model_family("FastVideo/LingBot-World-Base-Cam-Diffusers") == "lingbotworld"
    assert get_model_family("robbyant/lingbot-world-fast-diffusers") == "lingbotworld_fast"

    cfg = LingBotWorldActI2V480PConfig()
    assert cfg.boundary_ratio == 0.947           # dual-expert MoE boundary
    assert cfg.dit_config.arch_config.control_dim == 7
    assert cfg.dit_config.arch_config.in_channels == 36
    assert cfg.dit_config.arch_config.out_channels == 16


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
def test_lingbotworld_act_pipeline_smoke() -> None:
    """Load the real MoE pipeline and generate a short clip driven by act2cam."""
    if not _weights_available():
        pytest.skip(f"weights for {MODEL_ID} unavailable")

    from fastvideo import VideoGenerator
    from fastvideo.models.dits.lingbotworld.act_utils import prepare_action_embedding

    num_frames = 21
    c2ws_plucker_emb, num_frames = prepare_action_embedding(
        action_string="w-12,wl-8,none-4",
        num_frames=num_frames,
        height=480,
        width=832,
        spatial_scale=8,
    )

    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )

    result = generator.generate_video(
        "a calm river winding through a green valley under a bright sky",
        image_path="https://raw.githubusercontent.com/Robbyant/lingbot-world/main/examples/00/image.jpg",
        num_frames=num_frames,
        height=480,
        width=832,
        num_inference_steps=2,
        save_video=False,
        return_frames=True,
        c2ws_plucker_emb=c2ws_plucker_emb,
    )

    if isinstance(result, list):
        result = result[0]
    frames = np.asarray(result["frames"])
    assert frames.size > 0, "empty output"
    assert np.isfinite(frames).all(), "non-finite pixels in output"
    print(f"\n[smoke] frames shape={frames.shape} dtype={frames.dtype} "
          f"min={frames.min():.1f} max={frames.max():.1f} std={frames.std():.1f}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
