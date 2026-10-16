# SPDX-License-Identifier: Apache-2.0
from fastvideo.api.presets import InferencePreset, PresetStageSpec

FLUX3_KLEIN = InferencePreset(
    name="flux3_klein",
    version=1,
    model_family="flux3",
    description="FLUX.3 Image Klein with native prompt upsampling",
    workload_type="t2i",
    stage_schemas=(PresetStageSpec(
        name="denoise",
        kind="denoising",
        description="Distilled flow-map sampling",
        allowed_overrides=frozenset({"num_inference_steps", "flux3_alpha"}),
    ), ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "negative_prompt": "",
        "guidance_scale": 1.0,
        "num_inference_steps": 16,
        "flux3_alpha": 6.93,
        "flux3_auto_size": False,
        "flux3_upsample_prompt": True,
        "flux3_pup_max_new_tokens": 3584,
    },
)

ALL_PRESETS = (FLUX3_KLEIN, )
