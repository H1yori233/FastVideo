# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.layers.visual_embedding import TimestepEmbedder


def test_timestep_embedder_preserves_batch_for_sequence_timesteps() -> None:
    torch.manual_seed(0)
    embedder = TimestepEmbedder(
        hidden_size=12,
        frequency_embedding_size=8,
    )
    with torch.no_grad():
        for parameter in embedder.parameters():
            parameter.uniform_(-0.1, 0.1)
    timesteps = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    actual = embedder(timesteps.flatten(), timestep_seq_len=3)
    expected = embedder(timesteps.flatten()).unflatten(0, (2, 3))

    assert actual.shape == (2, 3, 12)
    torch.testing.assert_close(actual, expected)
