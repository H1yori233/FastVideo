# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class Flux3TextEncoderArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["Flux3TextEncoder"])
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    attention_bias: bool = False
    pad_token_id: int = 151643
    eos_token_id: int = 151645
    text_len: int = 4096
    pad_multiple: int = 80
    output_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
    output_hidden_states: bool = True
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.TORCH_SDPA, )
    param_names_mapping: dict = field(default_factory=lambda: {r"^model\.language_model\.(.*)$": r"\1"})


@dataclass
class Flux3TextEncoderConfig(TextEncoderConfig):
    arch_config: Flux3TextEncoderArchConfig = field(default_factory=Flux3TextEncoderArchConfig)
    is_chat_model: bool = True

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        source = dict(source_model_dict)
        source.update(source.pop("text_config", {}))
        rope = source.get("rope_parameters", source.get("rope_scaling", {}))
        if "mrope_section" in rope:
            source["mrope_section"] = tuple(rope["mrope_section"])
        if "rope_theta" in rope:
            source["rope_theta"] = rope["rope_theta"]
        source["architectures"] = ["Flux3TextEncoder"]
        super().update_model_arch(source)
