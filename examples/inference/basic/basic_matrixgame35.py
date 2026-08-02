# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 camera-conditioned I2V.

Required inputs:
    MATRIXGAME35_IMAGE=/path/to/input.png
    MATRIXGAME35_CAMERA=/path/to/camera.npz
    MATRIXGAME35_CAPTION=/path/to/caption.json  # or MATRIXGAME35_PROMPT

Select ``base-first`` (default), ``base-third``, or ``distilled`` with
``MATRIXGAME35_VARIANT``. Distilled additionally accepts
``MATRIXGAME35_PROFILE=standard|hiar-sde|sink-anchor-context``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OutputConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.configs.pipelines.matrixgame35 import MATRIXGAME35_DISTILLED_PROFILES

_MODEL_IDS = {
    "base-first": "FastVideo/Matrix-Game-3.5-Base-First-Person-Diffusers",
    "base-third": "FastVideo/Matrix-Game-3.5-Base-Third-Person-Diffusers",
    "distilled": "FastVideo/Matrix-Game-3.5-Distilled-First-Person-Diffusers",
}


def _required_path(name: str) -> str:
    value = os.getenv(name, "")
    if not value or not Path(value).is_file():
        raise SystemExit(f"{name} must point to an existing file")
    return value


def main() -> None:
    variant = os.getenv("MATRIXGAME35_VARIANT", "base-first")
    if variant not in _MODEL_IDS:
        raise SystemExit(f"MATRIXGAME35_VARIANT must be one of {tuple(_MODEL_IDS)}")

    profile = os.getenv("MATRIXGAME35_PROFILE", "standard")
    if profile not in MATRIXGAME35_DISTILLED_PROFILES:
        raise SystemExit(f"MATRIXGAME35_PROFILE must be one of {MATRIXGAME35_DISTILLED_PROFILES}")
    if variant != "distilled" and profile != "standard":
        raise SystemExit("MATRIXGAME35_PROFILE only applies to the distilled variant")

    image_path = _required_path("MATRIXGAME35_IMAGE")
    camera_path = _required_path("MATRIXGAME35_CAMERA")
    caption_path = os.getenv("MATRIXGAME35_CAPTION") or None
    prompt = os.getenv("MATRIXGAME35_PROMPT") or None
    if caption_path is not None and not Path(caption_path).is_file():
        raise SystemExit("MATRIXGAME35_CAPTION must point to an existing file")
    if caption_path is None and prompt is None:
        raise SystemExit("Set MATRIXGAME35_CAPTION or MATRIXGAME35_PROMPT")

    pipeline_overrides = ({"matrixgame35_distilled_profile": profile} if variant == "distilled" else {})
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=os.getenv("MATRIXGAME35_MODEL_PATH", _MODEL_IDS[variant]),
            pipeline=PipelineSelection(experimental=pipeline_overrides),
        ))

    guidance_scale = 1.0 if profile == "hiar-sde" else (3.0 if variant == "distilled" else 5.0)
    subject_refs = os.getenv("MATRIXGAME35_SUBJECT_REFS") or None
    if subject_refs is not None and variant != "base-third":
        raise SystemExit("MATRIXGAME35_SUBJECT_REFS is only supported by base-third")

    try:
        generator.generate(
            GenerationRequest(
                prompt=prompt,
                inputs=InputConfig(
                    image_path=image_path,
                    camera_trajectory=camera_path,
                    caption_path=caption_path,
                    subject_ref_source=subject_refs,
                ),
                sampling=SamplingConfig(
                    seed=3407,
                    height=704,
                    width=1280,
                    num_frames=int(os.getenv("MATRIXGAME35_NUM_FRAMES", "85")),
                    fps=16,
                    num_inference_steps=3 if variant == "distilled" else 25,
                    guidance_scale=guidance_scale,
                ),
                output=OutputConfig(
                    output_path=os.getenv("MATRIXGAME35_OUTPUT", "outputs/matrixgame35"),
                    output_video_name=f"matrixgame35_{variant}_{profile}.mp4",
                    save_video=True,
                    return_frames=False,
                ),
            ))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
