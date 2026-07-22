# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fastvideo.train.models.wan.wan_ti2v import WanTI2VModel


def test_ti2v_accepts_single_frame_conditioning_latent() -> None:
    model = object.__new__(WanTI2VModel)
    object.__setattr__(model, "transformer", SimpleNamespace(patch_size=(1, 2, 2)))
    object.__setattr__(
        model,
        "training_config",
        SimpleNamespace(pipeline_config=SimpleNamespace(expand_timesteps=True)),
    )

    hidden_states = torch.randn(2, 16, 9, 4, 6)
    first_frame_latent = torch.randn(2, 16, 1, 4, 6)
    timestep = torch.tensor([500.0, 750.0])

    conditioned, expanded_timestep = model._apply_first_frame_latent(
        hidden_states,
        timestep,
        first_frame_latent,
    )

    assert torch.equal(conditioned[:, :, :1], first_frame_latent)
    assert torch.equal(conditioned[:, :, 1:], hidden_states[:, :, 1:])
    assert expanded_timestep.shape == (2, 9 * 2 * 3)
    assert torch.count_nonzero(expanded_timestep[:, :2 * 3]) == 0
    assert torch.equal(expanded_timestep[0, 2 * 3:], torch.full((8 * 2 * 3,), 500.0))
