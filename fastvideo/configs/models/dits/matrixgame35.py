# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 transformer configuration.

The three released checkpoints share one Wan2.2-TI2V-5B backbone. Variant
differences are represented by ``subject_ref_memory_max_refs`` and ``causal``;
they do not require separate model classes.
"""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class MatrixGame35WanVideoArchConfig(WanVideoArchConfig):
    """Architecture shared by every public Matrix-Game 3.5 checkpoint."""

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(weight|bias)$": r"patch_embedding.proj.\1",
            r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^head\.head\.(.*)$": r"proj_out.\1",
            r"^head\.modulation$": r"scale_shift_table",
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
            r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
        })
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None

    use_prope: bool = True
    prope_attention_interval: int = 1
    prope_camera_layout: str = "full"
    prope_disable_native_rope: bool = False
    subject_ref_memory_max_refs: int = 0
    causal: bool = False
    causal_chunk_size: int = 3
    causal_window_size: int = 21

    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.TORCH_SDPA, )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prope_attention_interval != 1:
            raise ValueError("Released Matrix-Game 3.5 variants require PRoPE in every block.")
        if self.prope_camera_layout != "full":
            raise ValueError("Released Matrix-Game 3.5 variants require prope_camera_layout='full'.")
        if self.prope_disable_native_rope:
            raise ValueError("Released Matrix-Game 3.5 variants keep native 3D RoPE enabled.")
        if self.subject_ref_memory_max_refs not in (0, 2, 4):
            raise ValueError("subject_ref_memory_max_refs must be one of 0, 2, or 4.")


@dataclass
class MatrixGame35WanVideoConfig(WanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=MatrixGame35WanVideoArchConfig)
    prefix: str = "Wan"
