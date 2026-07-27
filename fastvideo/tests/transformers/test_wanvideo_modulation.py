# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.models.dits.wanvideo import _prepare_wan_modulation


def test_prepare_wan_modulation_supports_ti2v_tokenwise_temb():
    batch_size = 2
    seq_len = 7
    hidden_dim = 8
    scale_shift_table = torch.randn(1, 6, hidden_dim)
    temb = torch.randn(batch_size, seq_len, 6, hidden_dim)

    modulation = _prepare_wan_modulation(scale_shift_table, temb)

    assert len(modulation) == 6
    assert all(value.shape == (batch_size, seq_len, hidden_dim)
               for value in modulation)
    expected = scale_shift_table.unsqueeze(0) + temb.float()
    torch.testing.assert_close(torch.stack(modulation, dim=2), expected)


def test_prepare_wan_modulation_preserves_samplewise_temb():
    batch_size = 2
    hidden_dim = 8
    scale_shift_table = torch.randn(1, 6, hidden_dim)
    temb = torch.randn(batch_size, 6, hidden_dim)

    modulation = _prepare_wan_modulation(scale_shift_table, temb)

    assert len(modulation) == 6
    assert all(value.shape == (batch_size, 1, hidden_dim)
               for value in modulation)
    expected = scale_shift_table + temb.float()
    torch.testing.assert_close(torch.cat(modulation, dim=1), expected)
