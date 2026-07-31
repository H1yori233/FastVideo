# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.codec import (
    decode_matrixgame35_video,
    encode_matrixgame35_independent_frames,
    matrixgame35_memory_latents,
    matrixgame35_uint8_to_frames,
    matrixgame35_video_to_uint8,
)


class _Posterior:

    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class _FakeVAE:

    def __init__(self):
        self.config = SimpleNamespace(latents_mean=(1.0, 2.0), latents_std=(2.0, 4.0))
        self.encode_inputs = []
        self.last_decode = None

    def encode(self, video):
        self.encode_inputs.append(video.clone())
        value = torch.stack((video[:, 0], video[:, 1]), dim=1)
        return _Posterior(value)

    def decode(self, latents):
        self.last_decode = latents.clone()
        return torch.cat((latents[:, :1], latents[:, 1:2], latents[:, :1]), dim=1)


def test_independent_frames_use_sequential_single_frame_vae_calls_and_posterior_mode():
    vae = _FakeVAE()
    frames = torch.zeros(3, 3, 4, 5)
    frames[:, 0] = 5
    frames[:, 1] = 10

    latents = encode_matrixgame35_independent_frames(vae, frames)

    assert [value.shape for value in vae.encode_inputs] == [(1, 3, 1, 4, 5)] * 3
    assert latents.shape == (3, 2, 1, 4, 5)
    torch.testing.assert_close(latents[:, 0], torch.full((3, 1, 4, 5), 2.0))
    torch.testing.assert_close(latents[:, 1], torch.full((3, 1, 4, 5), 2.0))


def test_memory_layout_is_cpu_fp32_channel_first():
    vae = _FakeVAE()
    frames = torch.zeros(3, 3, 2, 4, dtype=torch.float16)

    memory = matrixgame35_memory_latents(vae, frames)

    assert memory.shape == (2, 3, 2, 4)
    assert memory.dtype == torch.float32
    assert memory.device.type == "cpu"


def test_decode_inverts_latent_statistics_before_vae():
    vae = _FakeVAE()
    latents = torch.zeros(1, 2, 1, 2, 2)

    video = decode_matrixgame35_video(vae, latents)

    torch.testing.assert_close(vae.last_decode[:, 0], torch.ones(1, 1, 2, 2))
    torch.testing.assert_close(vae.last_decode[:, 1], torch.full((1, 1, 2, 2), 2.0))
    assert video.min().item() == 1.0
    assert video.max().item() == 1.0


def test_uint8_conversion_matches_released_rounding_boundary():
    video = torch.tensor([[[[[0.0, 0.5]]], [[[1.0, 0.25]]], [[[0.1, 0.9]]]]])

    output = matrixgame35_video_to_uint8(video)

    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output[0, 0], np.array([[0, 255, 25], [127, 63, 229]], dtype=np.uint8))


def test_uint8_publication_round_trip_uses_normalized_rgb_frames():
    uint8 = np.array([[[[0, 127, 255]]]], dtype=np.uint8)

    frames = matrixgame35_uint8_to_frames(uint8, device="cpu")

    assert frames.shape == (1, 3, 1, 1)
    torch.testing.assert_close(frames[0, :, 0, 0], torch.tensor([-1.0, -1.0 / 255.0, 1.0]))


def test_codec_rejects_temporally_compressed_frame_input():
    with pytest.raises(ValueError, match=r"\[N,3,H,W\]"):
        encode_matrixgame35_independent_frames(_FakeVAE(), torch.zeros(1, 3, 2, 4, 5))
