# SPDX-License-Identifier: Apache-2.0
"""Shared component and pipeline configs for Matrix-Game 3.5."""

from collections.abc import Callable
from dataclasses import dataclass, field
import html
import math
import re

import ftfy
import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.matrixgame35 import (
    MatrixGame35WanVideoArchConfig,
    MatrixGame35WanVideoConfig,
)
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.encoders.matrixgame35 import MatrixGame35T5ArchConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.dreamx_world import (
    make_dreamx_world_5b_cam_vae_config, )
from fastvideo.configs.pipelines.wan import t5_postprocess_text


def make_matrixgame35_vae_config() -> WanVAEConfig:
    """Return the shared Wan2.2 48-channel VAE with both codec halves loaded."""

    config = make_dreamx_world_5b_cam_vae_config()
    config.load_encoder = True
    config.load_decoder = True
    return config


def make_matrixgame35_text_encoder_config() -> T5Config:
    """Return the UMT5-XXL config used by the official Matrix-Game runtime."""

    return T5Config(
        arch_config=MatrixGame35T5ArchConfig(
            vocab_size=256384,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_decoder_layers=None,
            num_heads=64,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            text_len=512,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=False,
        ),
        prefix="umt5",
    )


def matrixgame35_preprocess_text(prompt: str) -> str:
    """Apply the official Wan tokenizer's whitespace cleaning."""

    text = ftfy.fix_text(prompt)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


matrixgame35_postprocess_text = t5_postprocess_text

MATRIXGAME35_DISTILLED_PROFILES = ("standard", "hiar-sde", "sink-anchor-context")


def matrixgame35_distilled_profile_settings(profile: str, *, dynamic_context: bool = True) -> dict[str, bool | str]:
    """Translate the public profile into the released rollout policy switches."""
    if profile not in MATRIXGAME35_DISTILLED_PROFILES:
        raise ValueError(f"matrixgame35_distilled_profile must be one of {MATRIXGAME35_DISTILLED_PROFILES}, "
                         f"got {profile!r}.")
    return {
        "prefix_noise_mode": "hiar_sde" if profile == "hiar-sde" else "none",
        "noise_dynamic_context": profile == "hiar-sde" and bool(dynamic_context),
        "force_original_anchor": profile == "sink-anchor-context",
    }


def resolve_matrixgame35_hiar_scales(
    profile: str,
    scales: tuple[float, ...],
    *,
    num_steps: int,
) -> tuple[float, ...]:
    """Resolve the upstream empty-list default and validate per-step HiAR scales."""
    matrixgame35_distilled_profile_settings(profile)
    values = tuple(float(value) for value in scales)
    if profile != "hiar-sde":
        if values:
            raise ValueError("matrixgame35_distilled_hiar_scales is only valid for profile='hiar-sde'.")
        return ()
    if not values:
        return (1.0, ) * int(num_steps)
    if len(values) != int(num_steps):
        raise ValueError("matrixgame35_distilled_hiar_scales must contain one value per denoising step.")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("matrixgame35_distilled_hiar_scales values must be finite and lie in [0, 1].")
    return values


def make_matrixgame35_base_first_person_dit_config() -> MatrixGame35WanVideoConfig:
    """Return the Base first-person architecture, including its two-row ref table."""
    return MatrixGame35WanVideoConfig(arch_config=MatrixGame35WanVideoArchConfig(subject_ref_memory_max_refs=2))


def make_matrixgame35_base_third_person_dit_config() -> MatrixGame35WanVideoConfig:
    """Return the Base third-person architecture with four subject-reference rows."""
    return MatrixGame35WanVideoConfig(arch_config=MatrixGame35WanVideoArchConfig(subject_ref_memory_max_refs=4))


def make_matrixgame35_distilled_dit_config() -> MatrixGame35WanVideoConfig:
    """Return the released causal DiT with subject-reference memory disabled."""
    return MatrixGame35WanVideoConfig(arch_config=MatrixGame35WanVideoArchConfig(
        subject_ref_memory_max_refs=0,
        causal=True,
        causal_chunk_size=3,
        causal_window_size=21,
    ))


@dataclass
class MatrixGame35BaseFirstPersonPipelineConfig(PipelineConfig):
    """Released Base first-person STANDARD component and runtime config."""

    dit_config: DiTConfig = field(default_factory=make_matrixgame35_base_first_person_dit_config)
    vae_config: VAEConfig = field(default_factory=make_matrixgame35_vae_config)
    text_encoder_configs: tuple[EncoderConfig,
                                ...] = field(default_factory=lambda: (make_matrixgame35_text_encoder_config(), ))
    preprocess_text_funcs: tuple[Callable[[str], str],
                                 ...] = field(default_factory=lambda: (matrixgame35_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (matrixgame35_postprocess_text, ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    flow_shift: float | None = 5.0
    ti2v_task: bool = True
    is_causal: bool = False
    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "bf16"
    vae_decode_precision: str | None = "bf16"
    dit_precision: str = "bf16"
    matrixgame35_height: int = 704
    matrixgame35_width: int = 1280
    matrixgame35_da3_model_ref: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

    def __post_init__(self) -> None:
        if not isinstance(self.matrixgame35_da3_model_ref, str) or not self.matrixgame35_da3_model_ref.strip():
            raise ValueError("matrixgame35_da3_model_ref must be a non-empty model ID or local path.")
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class MatrixGame35BaseThirdPersonPipelineConfig(MatrixGame35BaseFirstPersonPipelineConfig):
    """Released Base third-person STANDARD component and runtime config."""

    dit_config: DiTConfig = field(default_factory=make_matrixgame35_base_third_person_dit_config)


@dataclass
class MatrixGame35DistilledFirstPersonPipelineConfig(MatrixGame35BaseFirstPersonPipelineConfig):
    """Released distilled first-person component and runtime-profile config."""

    dit_config: DiTConfig = field(default_factory=make_matrixgame35_distilled_dit_config)
    flow_shift: float | None = None
    is_causal: bool = True
    vae_tiling: bool = True
    matrixgame35_distilled_profile: str = "standard"
    matrixgame35_distilled_hiar_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        resolve_matrixgame35_hiar_scales(
            self.matrixgame35_distilled_profile,
            self.matrixgame35_distilled_hiar_scales,
            num_steps=3,
        )


__all__ = [
    "MATRIXGAME35_DISTILLED_PROFILES",
    "MatrixGame35BaseFirstPersonPipelineConfig",
    "MatrixGame35BaseThirdPersonPipelineConfig",
    "MatrixGame35DistilledFirstPersonPipelineConfig",
    "make_matrixgame35_base_first_person_dit_config",
    "make_matrixgame35_base_third_person_dit_config",
    "make_matrixgame35_distilled_dit_config",
    "make_matrixgame35_text_encoder_config",
    "make_matrixgame35_vae_config",
    "matrixgame35_distilled_profile_settings",
    "matrixgame35_postprocess_text",
    "matrixgame35_preprocess_text",
    "resolve_matrixgame35_hiar_scales",
]
