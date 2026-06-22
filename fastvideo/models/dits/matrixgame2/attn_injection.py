# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class _Injector:
    enabled: bool = False
    grid_rows: int = 30
    grid_cols: int = 52
    fg_box: tuple = (14, 30, 21, 31)  # (r0,r1,c0,c1) subject box in the token grid
    fg_boost: float = 4.0             # subject-anchor strength
    fg_sigma: tuple = (5.0, 5.0)      # (sigma_row, sigma_col) of the feather
    fg_qsupp: float = 6.0             # background query-suppression (enables travel)
    layer_lo: float = 0.2             # structural-layer band (fraction of depth)
    layer_hi: float = 0.8
    num_layers: int = 0

    def prepare(self, transformer) -> None:
        """Index every self-attention module in forward order."""
        from fastvideo.models.dits.matrixgame2.causal_model import (
            CausalMatrixGame2SelfAttention)
        attns = [m for m in transformer.modules()
                 if isinstance(m, CausalMatrixGame2SelfAttention)]
        for i, m in enumerate(attns):
            m._inj_layer_idx = i
        self.num_layers = len(attns)

    def on(self) -> None:
        self.enabled = True

    def off(self) -> None:
        self.enabled = False

    def maybe_apply(self, attn, q, k, v, sink_tokens):
        """Foreground-anchored attention output, or None to fall back to SDPA."""
        if not self.enabled or int(sink_tokens) <= 0:
            return None
        if self.num_layers > 0:  # gate to the structural middle layers
            frac = getattr(attn, "_inj_layer_idx", 0) / self.num_layers
            if not (self.layer_lo <= frac < self.layer_hi):
                return None
        return _fg_attention(q, k, v, int(sink_tokens), self.grid_rows,
                             self.grid_cols, self.fg_box, self.fg_boost,
                             self.fg_sigma, self.fg_qsupp)


def _fg_attention(q, k, v, sink_tokens, rows, cols, fg_box, fg_boost, sigma, qsupp):
    """SDPA with the feathered subject-anchor bias on the sink (frame-0) columns.

    q, k, v: [B, L, H, D]; the sink occupies the first ``sink_tokens`` columns of k.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [B,H,Lq,Lk]

    st, fpt = sink_tokens, rows * cols
    if st >= fpt and st % fpt == 0 and fg_boost != 0.0:
        r0, r1, c0, c1 = fg_box
        rc, cc = 0.5 * (r0 + r1), 0.5 * (c0 + c1)
        sr, sc = sigma
        idx = torch.arange(fpt, device=logits.device, dtype=torch.float32)
        w = torch.exp(-(((idx // cols - rc) ** 2) / (2 * sr * sr)
                        + ((idx % cols - cc) ** 2) / (2 * sc * sc)))  # [fpt] in (0,1]
        nf = st // fpt
        # zero-mean feather: subject columns boosted, periphery suppressed.
        bias = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        bias[:st] = ((w - w.mean()) * fg_boost).repeat(nf).to(logits.dtype)
        logits = logits + bias.view(1, 1, 1, -1)
        # background queries (low w) down-weight the sink -> the world travels.
        lq = logits.shape[2]
        if qsupp > 0.0 and lq % fpt == 0:
            qbias = -(1.0 - w.repeat(lq // fpt)) * qsupp     # [Lq], <=0
            sinkcol = torch.zeros(logits.shape[-1], device=logits.device,
                                  dtype=logits.dtype)
            sinkcol[:st] = 1.0
            logits = logits + (qbias.view(1, 1, lq, 1)
                               * sinkcol.view(1, 1, 1, -1)).to(logits.dtype)

    attn_w = torch.softmax(logits, dim=-1).to(vh.dtype)
    return torch.matmul(attn_w, vh).permute(0, 2, 1, 3)


# Module-level singleton used by the attention forward.
INJECTOR = _Injector()


def injected_sdpa(attn, q, k, v, sink_tokens):
    """Drop-in for the cache-attention SDPA call in causal_model.py."""
    out = INJECTOR.maybe_apply(attn, q, k, v, sink_tokens)
    if out is not None:
        return out
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
