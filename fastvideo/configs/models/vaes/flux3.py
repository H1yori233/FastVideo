# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
import re
from typing import Any

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Flux3VAEArchConfig(VAEArchConfig):
    embed_dim: int = 32
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    patch_size: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    bn_eps: float = 1e-4
    bn_momentum: float = 0.1
    scaling_factor: float = 1.0
    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 16
    param_names_mapping: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.latent_channels != self.embed_dim * self.patch_size**2:
            raise ValueError("latent_channels must equal embed_dim * patch_size**2")
        if self.spatial_compression_ratio != 2**(len(self.ch_mult) - 1) * self.patch_size:
            raise ValueError("spatial_compression_ratio must include VAE downsampling and patch packing")
        self.param_names_mapping = {
            r"^quant_conv\.(.*)$": r"encoder.quant_conv.\1",
            r"^post_quant_conv\.(.*)$": r"decoder.post_quant_conv.\1",
            r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)$": r"encoder.down.\1.block.\2.\3",
            r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.*)$": r"encoder.down.\1.downsample.\2",
            r"^(encoder|decoder)\.mid_block\.resnets\.0\.(.*)$": r"\1.mid.block_1.\2",
            r"^(encoder|decoder)\.mid_block\.resnets\.1\.(.*)$": r"\1.mid.block_2.\2",
            r"^(encoder|decoder)\.mid_block\.attentions\.0\.group_norm\.(.*)$": r"\1.mid.attn_1.norm.\2",
            r"^(encoder|decoder)\.mid_block\.attentions\.0\.to_(q|k|v)\.(.*)$": r"\1.mid.attn_1.\2.\3",
            r"^(encoder|decoder)\.mid_block\.attentions\.0\.to_out\.0\.(.*)$": r"\1.mid.attn_1.proj_out.\2",
        }
        for index in range(len(self.ch_mult)):
            level = len(self.ch_mult) - 1 - index
            self.param_names_mapping[rf"^decoder\.up_blocks\.{index}\.resnets\.(\d+)\.(.*)$"] = (
                rf"decoder.up.{level}.block.\1.\2")
            self.param_names_mapping[rf"^decoder\.up_blocks\.{index}\.upsamplers\.0\.(.*)$"] = (
                rf"decoder.up.{level}.upsample.\1")

    def map_official_key(self, key: str) -> str:
        key = key.replace(".conv_norm_out.", ".norm_out.").replace(".conv_shortcut.", ".nin_shortcut.")
        for pattern, replacement in self.param_names_mapping.items():
            if re.match(pattern, key):
                return re.sub(pattern, replacement, key)
        return key


@dataclass
class Flux3VAEConfig(VAEConfig):
    arch_config: Flux3VAEArchConfig = field(default_factory=Flux3VAEArchConfig)
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        if "block_out_channels" not in source_model_dict:
            super().update_model_arch(source_model_dict)
            return
        source = source_model_dict
        channels = source["block_out_channels"]
        patch = source["patch_size"]
        if len(patch) != 2 or patch[0] != patch[1]:
            raise ValueError("FLUX.3 VAE requires square latent patches")
        if source.get("decoder_block_out_channels") not in (None, channels):
            raise ValueError("FLUX.3 VAE requires matching encoder and decoder channels")
        expected = {
            "act_fn": "silu",
            "use_quant_conv": True,
            "use_post_quant_conv": True,
            "mid_block_add_attention": True,
            "down_block_types": ["DownEncoderBlock2D"] * len(channels),
            "up_block_types": ["UpDecoderBlock2D"] * len(channels),
        }
        for name, value in expected.items():
            if source.get(name, value) != value:
                raise ValueError(f"Unsupported FLUX.3 VAE {name}: {source[name]}")
        translated = dict(source)
        translated.update(
            embed_dim=source["latent_channels"],
            latent_channels=source["latent_channels"] * patch[0]**2,
            patch_size=patch[0],
            ch=channels[0],
            ch_mult=tuple(channel // channels[0] for channel in channels),
            num_res_blocks=source["layers_per_block"],
            bn_eps=source["batch_norm_eps"],
            bn_momentum=source["batch_norm_momentum"],
            spatial_compression_ratio=2**(len(channels) - 1) * patch[0],
        )
        super().update_model_arch(translated)
