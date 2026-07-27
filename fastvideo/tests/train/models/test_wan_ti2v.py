# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fastvideo.train.models.wan.wan import WanModel


def test_ti2v_accepts_single_frame_conditioning_latent() -> None:
    model = object.__new__(WanModel)
    object.__setattr__(model, "transformer", SimpleNamespace(patch_size=(1, 2, 2)))

    hidden_states = torch.randn(2, 16, 9, 4, 6)
    first_frame_latent = torch.randn(2, 16, 1, 4, 6)
    timestep = torch.tensor([500.0, 750.0])

    kwargs = model._build_distill_input_kwargs(
        hidden_states.permute(0, 2, 1, 3, 4),
        timestep,
        {
            "encoder_hidden_states": torch.empty(2, 1, 1),
            "encoder_attention_mask": torch.empty(2, 1),
            "first_frame_latent": first_frame_latent,
        },
    )
    conditioned = kwargs["hidden_states"]
    expanded_timestep = kwargs["timestep"]

    assert torch.equal(conditioned[:, :, :1], first_frame_latent)
    assert torch.equal(conditioned[:, :, 1:], hidden_states[:, :, 1:])
    assert expanded_timestep.shape == (2, 9 * 2 * 3)
    assert torch.count_nonzero(expanded_timestep[:, :2 * 3]) == 0
    assert torch.equal(expanded_timestep[0, 2 * 3:], torch.full((8 * 2 * 3,), 500.0))
