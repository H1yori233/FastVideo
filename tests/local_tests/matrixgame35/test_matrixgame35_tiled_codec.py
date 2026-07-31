# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.vaes.wanvae import AutoencoderKLWan
from fastvideo.pipelines.basic.matrixgame35.codec import (
    MATRIXGAME35_VAE_DECODE_TILE_SIZE,
    MATRIXGAME35_VAE_DECODE_TILE_STRIDE,
    MATRIXGAME35_VAE_ENCODE_TILE_SIZE,
    MATRIXGAME35_VAE_ENCODE_TILE_STRIDE,
    _matrixgame35_spatial_tile_tasks,
    decode_matrixgame35_tiled_video,
    encode_matrixgame35_tiled_video,
)


_ENCODE_LATENT_TASKS = (
    (0, 34, 0, 34),
    (0, 34, 16, 50),
    (0, 34, 32, 66),
    (0, 34, 48, 82),
    (18, 52, 0, 34),
    (18, 52, 16, 50),
    (18, 52, 32, 66),
    (18, 52, 48, 82),
)
_ENCODE_SAMPLE_TASKS = tuple(tuple(value * 16 for value in task) for task in _ENCODE_LATENT_TASKS)
_DECODE_LATENT_TASKS = (
    (0, 30, 0, 52),
    (0, 30, 26, 78),
    (0, 30, 52, 104),
    (15, 45, 0, 52),
    (15, 45, 26, 78),
    (15, 45, 52, 104),
)


class _Posterior:

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _FakeWanVAE:

    def __init__(
        self,
        *,
        encode_values: tuple[float, ...] = (),
        decode_values: tuple[float, ...] = (),
    ) -> None:
        self.config = SimpleNamespace(
            latents_mean=(0.0, ),
            latents_std=(1.0, ),
            scale_factor_spatial=16,
        )
        self.z_dim = 1
        self.encode_values = encode_values
        self.decode_values = decode_values
        self.encode_shapes: list[tuple[int, ...]] = []
        self.decode_shapes: list[tuple[int, ...]] = []

    def encode(self, video: torch.Tensor) -> _Posterior:
        call_index = len(self.encode_shapes)
        self.encode_shapes.append(tuple(video.shape))
        value = self.encode_values[call_index] if self.encode_values else 0.0
        batch, _, frames, height, width = video.shape
        latents = torch.full(
            (batch, 1, (frames + 3) // 4, height // 16, width // 16),
            value,
            dtype=video.dtype,
            device=video.device,
        )
        return _Posterior(latents)

    def decode_unclamped(self, latents: torch.Tensor) -> torch.Tensor:
        call_index = len(self.decode_shapes)
        self.decode_shapes.append(tuple(latents.shape))
        value = self.decode_values[call_index] if self.decode_values else 0.0
        batch, _, frames, height, width = latents.shape
        return torch.full(
            (batch, 3, (frames - 1) * 4 + 1, height * 16, width * 16),
            value,
            dtype=latents.dtype,
            device=latents.device,
        )


def _axis_mask(length: int, lower_bound: bool, upper_bound: bool, border_width: int) -> torch.Tensor:
    mask = torch.ones(length)
    ramp = (torch.arange(border_width) + 1) / border_width
    if not lower_bound:
        mask[:border_width] = ramp
    if not upper_bound:
        mask[-border_width:] = torch.flip(ramp, dims=(0, ))
    return mask


def _reference_constant_tile_merge(
    *,
    tasks: tuple[tuple[int, int, int, int], ...],
    constants: tuple[float, ...],
    input_height: int,
    input_width: int,
    output_scale: int,
    border_width: tuple[int, int],
    channels: int,
) -> torch.Tensor:
    output_height = input_height * output_scale
    output_width = input_width * output_scale
    values = torch.zeros(1, channels, 1, output_height, output_width)
    weight = torch.zeros(1, 1, 1, output_height, output_width)
    for constant, (h, h_end, w, w_end) in zip(constants, tasks, strict=True):
        tile_height = (min(h_end, input_height) - h) * output_scale
        tile_width = (min(w_end, input_width) - w) * output_scale
        mask_h = _axis_mask(
            tile_height,
            h == 0,
            h_end >= input_height,
            border_width[0],
        ).view(tile_height, 1)
        mask_w = _axis_mask(
            tile_width,
            w == 0,
            w_end >= input_width,
            border_width[1],
        ).view(1, tile_width)
        mask = torch.minimum(mask_h, mask_w).view(1, 1, 1, tile_height, tile_width)
        target_h = h * output_scale
        target_w = w * output_scale
        values[:, :, :, target_h:target_h + tile_height, target_w:target_w + tile_width] += constant * mask
        weight[:, :, :, target_h:target_h + tile_height, target_w:target_w + tile_width] += mask
    return values / weight


def test_matrixgame35_released_tile_starts_skip_redundant_tail_tiles() -> None:
    assert _matrixgame35_spatial_tile_tasks(
        44,
        80,
        MATRIXGAME35_VAE_DECODE_TILE_SIZE,
        MATRIXGAME35_VAE_DECODE_TILE_STRIDE,
    ) == _DECODE_LATENT_TASKS
    assert _matrixgame35_spatial_tile_tasks(
        704,
        1280,
        tuple(value * 16 for value in MATRIXGAME35_VAE_ENCODE_TILE_SIZE),
        tuple(value * 16 for value in MATRIXGAME35_VAE_ENCODE_TILE_STRIDE),
    ) == _ENCODE_SAMPLE_TASKS


def test_matrixgame35_tiled_encode_uses_eight_tiles_at_release_resolution() -> None:
    vae = _FakeWanVAE()
    encoded = encode_matrixgame35_tiled_video(vae, torch.zeros(1, 3, 1, 704, 1280))

    assert encoded.shape == (1, 1, 1, 44, 80)
    assert len(vae.encode_shapes) == 8
    assert vae.encode_shapes == [
        (1, 3, 1, 544, 544),
        (1, 3, 1, 544, 544),
        (1, 3, 1, 544, 544),
        (1, 3, 1, 544, 512),
        (1, 3, 1, 416, 544),
        (1, 3, 1, 416, 544),
        (1, 3, 1, 416, 544),
        (1, 3, 1, 416, 512),
    ]


def test_matrixgame35_tiled_decode_uses_six_tiles_at_release_resolution() -> None:
    vae = _FakeWanVAE()
    decoded = decode_matrixgame35_tiled_video(vae, torch.zeros(1, 1, 1, 44, 80))

    assert decoded.shape == (1, 3, 1, 704, 1280)
    assert len(vae.decode_shapes) == 6
    assert vae.decode_shapes == [
        (1, 1, 1, 30, 52),
        (1, 1, 1, 30, 52),
        (1, 1, 1, 30, 28),
        (1, 1, 1, 29, 52),
        (1, 1, 1, 29, 52),
        (1, 1, 1, 29, 28),
    ]


def test_matrixgame35_tiled_encode_matches_weighted_mask_merge() -> None:
    constants = (0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0)
    vae = _FakeWanVAE(encode_values=constants)
    encoded = encode_matrixgame35_tiled_video(vae, torch.zeros(1, 3, 1, 704, 1280))
    expected = _reference_constant_tile_merge(
        tasks=_ENCODE_LATENT_TASKS,
        constants=constants,
        input_height=44,
        input_width=80,
        output_scale=1,
        border_width=(16, 18),
        channels=1,
    )

    assert_close(encoded, expected, atol=0.0, rtol=0.0)


def test_matrixgame35_tiled_decode_matches_weighted_merge_and_clamps_afterward() -> None:
    constants = (2.0, 0.0, -0.5, 0.25, 0.5, 0.75)
    vae = _FakeWanVAE(decode_values=constants)
    decoded = decode_matrixgame35_tiled_video(vae, torch.zeros(1, 1, 1, 44, 80))
    raw = _reference_constant_tile_merge(
        tasks=_DECODE_LATENT_TASKS,
        constants=constants,
        input_height=44,
        input_width=80,
        output_scale=16,
        border_width=(240, 416),
        channels=3,
    )
    expected = raw.clamp(-1.0, 1.0).mul(0.5).add(0.5)

    assert_close(decoded, expected, atol=0.0, rtol=0.0)
    assert decoded[0, 0, 0, 0, 416].item() == 1.0
    preclamped_tiles = list(constants)
    preclamped_tiles[0] = 1.0
    incorrectly_preclamped = _reference_constant_tile_merge(
        tasks=_DECODE_LATENT_TASKS,
        constants=tuple(preclamped_tiles),
        input_height=44,
        input_width=80,
        output_scale=16,
        border_width=(240, 416),
        channels=3,
    ).mul(0.5).add(0.5)
    assert incorrectly_preclamped[0, 0, 0, 0, 416].item() < 1.0


def test_matrixgame35_tiled_codec_preserves_temporal_and_spatial_shapes() -> None:
    vae = _FakeWanVAE()
    encoded = encode_matrixgame35_tiled_video(vae, torch.zeros(1, 3, 5, 32, 48))
    decoded = decode_matrixgame35_tiled_video(vae, torch.zeros(1, 1, 4, 2, 3))

    assert encoded.shape == (1, 1, 2, 2, 3)
    assert decoded.shape == (1, 3, 13, 32, 48)


@pytest.mark.parametrize(
    ("operation", "value"),
    (
        (encode_matrixgame35_tiled_video, torch.zeros(2, 3, 1, 32, 48)),
        (decode_matrixgame35_tiled_video, torch.zeros(2, 1, 1, 2, 3)),
    ),
)
def test_matrixgame35_tiled_codec_rejects_batched_inputs(operation, value) -> None:
    with pytest.raises(ValueError, match=r"\[1,"):
        operation(_FakeWanVAE(), value)


class _ScaleDecoder(torch.nn.Module):

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


class _TinyFeatureCacheWanVAE(AutoencoderKLWan):

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.use_feature_cache = True
        self.config = SimpleNamespace(patch_size=None, use_light_vae=False)
        self.post_quant_conv = torch.nn.Identity()
        self.decoder = _ScaleDecoder()
        self._feat_map = []
        self._conv_idx = 0

    def clear_cache(self) -> None:
        self._feat_map = []
        self._conv_idx = 0


def test_wan_feature_cache_unclamped_decode_preserves_public_decode_semantics() -> None:
    vae = _TinyFeatureCacheWanVAE()
    latent = torch.full((1, 1, 2, 1, 1), 0.75)

    raw = vae.decode_unclamped(latent)
    public = vae.decode(latent)

    assert_close(raw, torch.full_like(raw, 1.5))
    assert public.dtype == torch.float32
    assert_close(public, torch.ones_like(public))
