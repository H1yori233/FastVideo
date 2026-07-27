# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fastvideo.train.models.wan.wan import WanModel


def test_ti2v_conditions_first_frame() -> None:
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
    assert torch.equal(kwargs["hidden_states"][:, :, :1], first_frame_latent)
    assert torch.count_nonzero(kwargs["timestep"][:, :6]) == 0
    assert torch.equal(kwargs["timestep"][:, 6:],
                       timestep[:, None].expand(-1, 48))
