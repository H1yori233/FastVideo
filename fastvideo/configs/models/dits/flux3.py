# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
import re

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_flux3_block(name: str, module) -> bool:
    return (name.startswith(("content_mode_blocks.", "txt_mode_blocks.", "single_blocks."))
            and name.rsplit(".", 1)[-1].isdigit())


def _parameter_mapping() -> dict[str, str]:
    mapping = {
        r"^context_embedder\.weight$": "txt_in.weight",
        r"^x_embedders\.(image|image_cond)\.weight$": r"emb_in.\1.weight",
        r"^timestep_embedder\.linear_1\.weight$": "time_in.in_layer.weight",
        r"^timestep_embedder\.linear_2\.weight$": "time_in.out_layer.weight",
        r"^target_timestep_embedder\.linear_1\.weight$": "ref_time_in.in_layer.weight",
        r"^target_timestep_embedder\.linear_2\.weight$": "ref_time_in.out_layer.weight",
        r"^(early|single)_stream_modulations\.(image|image_cond|txt)\.linear\.weight$":
        r"\1_stream_modulations.\2.lin.weight",
        r"^norm_out\.linear\.weight$": "final_layer.image.adaLN_modulation.1.weight",
        r"^proj_out\.weight$": "final_layer.image.linear.weight",
    }
    projections = {
        "attn.norm_q.weight": "norm.query_norm.scale",
        "attn.norm_k.weight": "norm.key_norm.scale",
        "attn.to_q.weight": "q_proj.weight",
        "attn.to_k.weight": "k_proj.weight",
        "attn.to_v.weight": "v_proj.weight",
        "attn.to_out.0.weight": "attn_out.weight",
        "ff.linear_in.weight": "mlp_in.weight",
        "ff.linear_out.weight": "mlp_out.weight",
    }
    for source, target in (("mode_blocks.image", "content_mode_blocks.image"), ("mode_blocks.image_cond",
                                                                                "content_mode_blocks.image_cond"),
                           ("txt_mode_blocks", "txt_mode_blocks"), ("single_transformer_blocks", "single_blocks")):
        for old, new in projections.items():
            mapping[rf"^{re.escape(source)}\.(\d+)\.{re.escape(old)}$"] = target + r".\1." + new
    return mapping


@dataclass
class Flux3ArchConfig(DiTArchConfig):
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    in_channels: int = 128
    cond_in_channels: int = 128
    joint_attention_dim: int = 20480
    num_layers: int = 5
    num_single_layers: int = 28
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 10000
    timestep_channels: int = 256
    eps: float = 1e-6
    flow_map: bool = True
    cast_prompt_embeds_to_dit_dtype: bool = True
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_flux3_block])
    param_names_mapping: dict = field(default_factory=_parameter_mapping)
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.TORCH_SDPA, )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels
        self.out_channels = self.in_channels
        if sum(self.axes_dims_rope) != self.attention_head_dim or any(d % 2 for d in self.axes_dims_rope):
            raise ValueError("RoPE axes must be even and sum to the attention head dimension")


@dataclass
class Flux3Config(DiTConfig):
    arch_config: Flux3ArchConfig = field(default_factory=Flux3ArchConfig)
    prefix: str = "Flux3"
