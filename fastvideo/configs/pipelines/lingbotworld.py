# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig
from fastvideo.configs.pipelines.wan import Wan2_2_I2V_A14B_Config, WanT2V480PConfig
from fastvideo.configs.models.dits.lingbotworld import (CausalLingBotWorldVideoConfig, LingBotWorldVideoConfig)


@dataclass
class LingBotWorldI2V480PConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorldVideoConfig)
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947


@dataclass
class LingBotWorldFastI2V480PConfig(WanT2V480PConfig):
    # Single block-causal DMD transformer (no MoE, no CLIP image encoder). Image
    # conditioning is the Wan-2.1 channel concat done in the denoising stage.
    dit_config: DiTConfig = field(default_factory=CausalLingBotWorldVideoConfig)
    is_causal: bool = True
    flow_shift: float | None = 3.0
    num_frames_per_block: int = 3
    context_noise: int = 0
    warp_denoising_step: bool = False
    boundary_ratio: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
