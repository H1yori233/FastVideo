# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field, fields
from typing import Any

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class Flux3PupTextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    mamba_ssm_dtype: str = "float32"
    max_position_embeddings: int = 262144
    pad_token_id: int | None = None
    eos_token_id: int = 248044
    bos_token_id: int | None = None
    use_cache: bool = True
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        })
    layer_types: list[str] = field(default_factory=list)
    model_type: str = "qwen3_5_text"

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class Flux3PupVisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    model_type: str = "qwen3_5_vision"


@dataclass
class Flux3PromptUpsamplerArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["Flux3PromptUpsampler"])
    text_config: Flux3PupTextConfig | dict = field(default_factory=Flux3PupTextConfig)
    vision_config: Flux3PupVisionConfig | dict = field(default_factory=Flux3PupVisionConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    pad_token_id: int = 248044
    eos_token_id: int = 248046
    text_len: int = 8192
    require_processor: bool = True
    param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.TORCH_SDPA, )

    def __post_init__(self):
        super().__post_init__()
        for name, cls in (("text_config", Flux3PupTextConfig), ("vision_config", Flux3PupVisionConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                accepted = {item.name for item in fields(cls)}
                setattr(self, name, cls(**{key: val for key, val in value.items() if key in accepted}))
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size


@dataclass
class Flux3PromptUpsamplerConfig(TextEncoderConfig):
    arch_config: Flux3PromptUpsamplerArchConfig = field(default_factory=Flux3PromptUpsamplerArchConfig)
    is_chat_model: bool = True

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        super().update_model_arch(source_model_dict | {"architectures": ["Flux3PromptUpsampler"]})
