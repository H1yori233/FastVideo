# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 inference presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec
from fastvideo.pipelines.basic.matrixgame35.prompts import MATRIXGAME35_NEGATIVE_PROMPT

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Camera-conditioned block rollout",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

_BASE_DEFAULTS = {
    "height": 704,
    "width": 1280,
    "num_frames": 85,
    "fps": 16,
    "guidance_scale": 5.0,
    "num_inference_steps": 25,
    "negative_prompt": MATRIXGAME35_NEGATIVE_PROMPT,
    "seed": 3407,
}

MATRIXGAME35_BASE_FIRST_PERSON = InferencePreset(
    name="matrixgame35_base_first_person",
    version=1,
    model_family="matrixgame35",
    description="Matrix-Game 3.5 Base first-person camera-conditioned I2V",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={**_BASE_DEFAULTS},
)

MATRIXGAME35_BASE_THIRD_PERSON = InferencePreset(
    name="matrixgame35_base_third_person",
    version=1,
    model_family="matrixgame35",
    description="Matrix-Game 3.5 Base third-person camera-conditioned I2V with optional subject references",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={**_BASE_DEFAULTS},
)

MATRIXGAME35_DISTILLED_FIRST_PERSON = InferencePreset(
    name="matrixgame35_distilled_first_person",
    version=1,
    model_family="matrixgame35",
    description="Matrix-Game 3.5 Distilled first-person camera-conditioned I2V",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        **_BASE_DEFAULTS,
        "guidance_scale": 3.0,
        "num_inference_steps": 3,
    },
)

ALL_PRESETS = (
    MATRIXGAME35_BASE_FIRST_PERSON,
    MATRIXGAME35_BASE_THIRD_PERSON,
    MATRIXGAME35_DISTILLED_FIRST_PERSON,
)
