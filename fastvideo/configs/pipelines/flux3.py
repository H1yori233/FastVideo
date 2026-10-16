# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.flux3 import Flux3Config
from fastvideo.configs.models.encoders.flux3 import Flux3TextEncoderConfig
from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig
from fastvideo.configs.models.vaes.flux3 import Flux3VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, postprocess_text, preprocess_text


@dataclass
class Flux3PipelineConfig(PipelineConfig):
    dit_config: Flux3Config = field(default_factory=Flux3Config)
    dit_precision: str = "bf16"
    vae_config: Flux3VAEConfig = field(default_factory=Flux3VAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = False
    vae_sp: bool = False
    text_encoder_configs: tuple = field(
        default_factory=lambda: (Flux3TextEncoderConfig(), Flux3PromptUpsamplerConfig()))
    text_encoder_precisions: tuple[str, ...] = ("bf16", "bf16")
    preprocess_text_funcs: tuple = (preprocess_text, preprocess_text)
    postprocess_text_funcs: tuple = (postprocess_text, postprocess_text)
    embedded_cfg_scale: float = 1.0
    scheduler_step_in_fp32: bool = True
