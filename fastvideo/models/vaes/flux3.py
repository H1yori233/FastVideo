# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo config and component loading.
"""Image autoencoder with channel-major 2x2 packing and (t, h, w, l) ids."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from fastvideo.configs.models.vaes.flux3 import Flux3VAEArchConfig, Flux3VAEConfig

DOWNSAMPLING_RATIO = Flux3VAEArchConfig().spatial_compression_ratio
LATENT_CHANNELS = Flux3VAEArchConfig().latent_channels


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def normalize(channels: int, config: Flux3VAEArchConfig) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=config.norm_num_groups, num_channels=channels, eps=config.norm_eps, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: Flux3VAEArchConfig):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = normalize(in_channels, config)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels, config)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int, config: Flux3VAEArchConfig):
        super().__init__()
        self.norm = normalize(channels, config)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        b, c, height, width = h.shape
        flat = lambda t: t.reshape(b, c, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        h = F.scaled_dot_product_attention(flat(self.q(h)), flat(self.k(h)), flat(self.v(h)))
        return x + self.proj_out(h.squeeze(1).permute(0, 2, 1).reshape(b, c, height, width))


class Encoder(nn.Module):
    def __init__(self, config: Flux3VAEArchConfig):
        super().__init__()
        self.num_levels = len(config.ch_mult)
        self.quant_conv = nn.Conv2d(2 * config.embed_dim, 2 * config.embed_dim, 1)
        self.conv_in = nn.Conv2d(config.in_channels, config.ch, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = config.ch
        for level, mult in enumerate(config.ch_mult):
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                down.block.append(ResnetBlock(block_in, config.ch * mult, config))
                block_in = config.ch * mult
            if level != self.num_levels - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, config)
        self.mid.attn_1 = AttnBlock(block_in, config)
        self.mid.block_2 = ResnetBlock(block_in, block_in, config)
        self.norm_out = normalize(block_in, config)
        self.conv_out = nn.Conv2d(block_in, 2 * config.embed_dim, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if level != self.num_levels - 1:
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))


class Decoder(nn.Module):
    def __init__(self, config: Flux3VAEArchConfig):
        super().__init__()
        self.num_levels = len(config.ch_mult)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.embed_dim, 1)
        block_in = config.ch * config.ch_mult[-1]
        self.conv_in = nn.Conv2d(config.embed_dim, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, config)
        self.mid.attn_1 = AttnBlock(block_in, config)
        self.mid.block_2 = ResnetBlock(block_in, block_in, config)
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            for _ in range(config.num_res_blocks + 1):
                up.block.append(ResnetBlock(block_in, config.ch * config.ch_mult[level], config))
                block_in = config.ch * config.ch_mult[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = normalize(block_in, config)
        self.conv_out = nn.Conv2d(block_in, config.out_channels, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(self.num_levels)):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))


class Flux3AutoEncoder(nn.Module):
    def __init__(self, config: Flux3VAEConfig | None = None):
        super().__init__()
        self.config = config or Flux3VAEConfig()
        arch = self.config.arch_config
        self.encoder = Encoder(arch) if self.config.load_encoder else None
        self.decoder = Decoder(arch) if self.config.load_decoder else None
        self.bn_eps = arch.bn_eps
        self.bn = nn.BatchNorm2d(arch.latent_channels, eps=self.bn_eps, momentum=arch.bn_momentum,
                                affine=False, track_running_stats=True)

    @classmethod
    def from_pretrained(cls, model_path, config=None, dtype=torch.float32, device="cpu"):
        path = Path(model_path)
        config = config or Flux3VAEConfig()
        config.update_model_arch(json.loads((path / "config.json").read_text()))
        with torch.device("meta"):
            model = cls(config)
        model.load_state_dict(load_file(str(path / "diffusion_pytorch_model.safetensors")), strict=True, assign=True)
        return model.to(device=device, dtype=dtype).eval()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = {}
        for name, tensor in state_dict.items():
            target = self.config.arch_config.map_official_key(name)
            if target in mapped:
                raise ValueError(f"Duplicate VAE checkpoint key: {target}")
            # Diffusers stores the 1x1 attention projections as linear weights.
            if ".mid.attn_1." in target and target.endswith(".weight") and tensor.ndim == 2:
                tensor = tensor[:, :, None, None]
            mapped[target] = tensor
        if strict:
            expected = self.state_dict().keys()
            missing, unexpected = sorted(expected - mapped.keys()), sorted(mapped.keys() - expected)
            if missing or unexpected:
                raise RuntimeError(f"VAE checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return super().load_state_dict(mapped, strict=strict, assign=assign)

    @torch.inference_mode()
    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """CHW image in [-1, 1] to tokens and (0, row, col, 0) ids."""
        if self.encoder is None:
            raise RuntimeError("VAE encoder is not loaded")
        arch = self.config.arch_config
        p = arch.patch_size
        mean, _ = self.encoder(image.unsqueeze(0)).chunk(2, dim=1)
        z = mean.reshape(1, arch.embed_dim, mean.shape[2] // p, p, mean.shape[3] // p, p)
        # Channel-major packing: (C, patch row, patch column).
        z = z.permute(0, 1, 3, 5, 2, 4)
        z = self.bn(z.reshape(1, arch.latent_channels, mean.shape[2] // p, mean.shape[3] // p))
        h, w = z.shape[2:]
        return z[0].permute(1, 2, 0).reshape(h * w, arch.latent_channels), latent_ids(h, w, z.device)

    @torch.inference_mode()
    def decode(self, tokens: Tensor, ids: Tensor) -> Tensor:
        """Tokens in row-major id order to a CHW image in [-1, 1]."""
        if self.decoder is None:
            raise RuntimeError("VAE decoder is not loaded")
        arch = self.config.arch_config
        p = arch.patch_size
        h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
        z = tokens.float().reshape(h, w, arch.latent_channels).permute(2, 0, 1).unsqueeze(0)
        z = z * torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps) + self.bn.running_mean.view(1, -1, 1, 1)
        z = z.reshape(1, arch.embed_dim, p, p, h, w).permute(0, 1, 4, 2, 5, 3)
        z = z.reshape(1, arch.embed_dim, p * h, p * w)
        return self.decoder(z)[0]


def latent_ids(h: int, w: int, device: torch.device | str = "cpu") -> Tensor:
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)


def latent_shape(height: int, width: int) -> tuple[int, int]:
    if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO


EntryClass = Flux3AutoEncoder
