# SPDX-License-Identifier: Apache-2.0
"""Cross-frame attention injection for Matrix-Game 2 causal inference.

Algorithm (image-driven editing of an autoregressive world model):

    A well-calibrated *source* rollout M(I_s, a) carries correct dynamics but
    the wrong appearance.  A *target* rollout M(I_e, a) on an OOD anchor has the
    right appearance but degraded dynamics.  Dynamics live in the **cross-frame
    attention routing** (how the current frame attends the KV-cache of past
    frames); appearance lives in the values V and in within-frame routing.

    So we transplant *only* the source's cross-frame routing into the target,
    in logit space:

        z_eff[:, cross] = (1 - lambda) * z_t[:, cross] + lambda * z_s[:, cross]

    where z = q k^T / sqrt(d).  This is a product-of-experts / geometric blend
    of the two routing distributions, always a valid distribution after softmax.
    The sink columns and the within-frame columns are left untouched (target),
    and V stays the target's V.  At lambda=1 the target content is routed
    exactly as the source moved it; appearance is never overwritten.

    `lambda` decays over denoising steps (structure is committed early under the
    4-step DMD schedule) and is gated to the structural middle layers.

Realization: the source and target rollouts share actions, seed and schedule,
so their sequence of attention calls is identical.  We RECORD (q_s, k_s_cross)
in call order during the source pass, then REPLAY them in order during the
target pass.  No cross-call keying, no pipeline batching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fastvideo.forward_context import get_forward_context


@dataclass
class _Injector:
    mode: str = "off"  # "off" | "record" | "replay"
    # Validated default: step-0-only at lambda 0.3 corrects OOD camera dynamics
    # toward the calibrated source while preserving appearance (no ghosting).
    # Injecting later steps (1..) couples motion transfer with source-appearance
    # bleed, so keep injection in step 0 (which sets structure under DMD).
    lambda_max: float = 0.3
    lambda_within: float = 0.0  # within-frame (pose/structure) blend strength
    lambda_v: float = 0.0       # value (appearance) hold strength for anchor_hold
    inject_steps: tuple[int, ...] = (0,)  # denoising steps that get injection
    layer_lo: float = 0.25  # structural-layer band (fraction of depth)
    layer_hi: float = 0.85
    num_layers: int = 0
    # foreground-sink: anchor only the center (subject) tokens of the sink frame,
    # so background/viewpoint can travel. grid + center box in token-grid coords.
    grid_rows: int = 30
    grid_cols: int = 52
    fg_box: tuple = (6, 24, 10, 42)  # (r0,r1,c0,c1) center kept; rest masked
    fg_boost: float = 0.0  # logit bias added to kept center sink cols (stronger ID)
    fg_soft: bool = False  # feathered gaussian bias on sink cols (no hard seam)
    fg_sigma: tuple = (6.0, 10.0)  # (sigma_row, sigma_col) for the feather
    fg_qsupp: float = 0.0  # soft: background queries down-weight the sink (travel)
    fg_boost_ramp: float = 0.0  # per-frame boost increase (counter tail drift)
    fg_boost_max: float = 0.0   # clamp for ramped boost (0 = no clamp)
    # frame-scheduled query-suppression decay: late in the rollout, ease off the
    # bg-query suppression so background queries RE-ATTEND the frame-0 background
    # appearance -> re-grounds bg style (fights Zelda-content hallucination)
    # without a hard geometric tether. qsupp_eff = max(floor, qsupp*(1-decay*f)).
    fg_qsupp_decay: float = 0.0  # per-latent-frame fractional decay of qsupp
    fg_qsupp_floor: float = 0.0  # absolute floor for decayed qsupp
    # subject-region VALUE anchor: blend the current within-frame values toward the
    # frame-0 anchor values, ONLY inside the subject box -> pins subject appearance/
    # identity (and pulls the glider-region back to the glider-free anchor) while
    # leaving routing/articulation and the background free. lambda in [0,1].
    fg_vhold: float = 0.0
    fg_vhold_adain: bool = False  # AdaIN (per-channel stat match) instead of blend

    _buffer: list = field(default_factory=list)
    _ptr: int = 0
    _anchor_v: dict = field(default_factory=dict)  # layer_idx -> first-block within V

    # ---- lifecycle -------------------------------------------------------
    def prepare(self, transformer) -> None:
        """Index every self-attention module in forward order."""
        from fastvideo.models.dits.matrixgame2.causal_model import (
            CausalMatrixGame2SelfAttention)
        attns = [
            m for m in transformer.modules()
            if isinstance(m, CausalMatrixGame2SelfAttention)
        ]
        for i, m in enumerate(attns):
            m._inj_layer_idx = i
        self.num_layers = len(attns)

    def start_fg_sink(self) -> None:
        """Anchor only the center (subject) tokens of the sink frame; let the
        peripheral (background) sink tokens be ignored so the viewpoint travels."""
        self.mode = "fg_sink"

    def start_anchor_hold(self) -> None:
        """Single-pass appearance hold: capture first-block within-V, then blend
        each later frame's within-V toward it (structure/motion stays free)."""
        self.mode = "anchor_hold"
        self._anchor_v = {}

    def start_adain_hold(self) -> None:
        """Let structure/articulation evolve freely, but re-normalize each frame's
        within-V per-channel statistics (mu,sigma) to the robot anchor's. Swaps
        identity (style) while preserving articulation (token-wise structure)."""
        self.mode = "adain_hold"
        self._anchor_v = {}

    def start_record(self) -> None:
        self.mode = "record"
        self._buffer = []
        self._ptr = 0

    def start_replay(self) -> None:
        self.mode = "replay"
        self._ptr = 0

    def off(self) -> None:
        self.mode = "off"

    # ---- per-call schedule ----------------------------------------------
    def _gate(self, layer_idx: int) -> float:
        """Step/layer gate with step-decay; 0 if this call isn't injected."""
        step = int(get_forward_context().current_timestep)
        if step not in self.inject_steps:
            return 0.0
        if self.num_layers > 0:
            frac = layer_idx / self.num_layers
            if not (self.layer_lo <= frac < self.layer_hi):
                return 0.0
        order = self.inject_steps.index(step)
        return 1.0 - order / max(1, len(self.inject_steps))

    # ---- the hook --------------------------------------------------------
    def maybe_apply(self, attn, q, k, v, sink_tokens, num_new_tokens,
                    start_frame=0):
        """Return blended attention output, or None to fall back to SDPA.

        q:        [B, Lq, H, D]  current-chunk queries (RoPE'd)
        k, v:     [B, Lk, H, D]  attention window = [sink | cross | within]
        cross columns = [sink_tokens : Lk - num_new_tokens]
        """
        if self.mode == "off":
            return None

        lk = k.shape[1]
        c0 = int(sink_tokens)
        c1 = lk - int(num_new_tokens)
        has_cross = c1 > c0
        layer_idx = getattr(attn, "_inj_layer_idx", 0)

        if self.mode == "fg_sink":
            if c0 <= 0:
                return None  # no sink retained this call -> ordinary SDPA
            if self.num_layers > 0:  # gate to structural layers (less haze)
                frac = layer_idx / self.num_layers
                if not (self.layer_lo <= frac < self.layer_hi):
                    return None
            # frame-scheduled boost: ramp up over the rollout to counter the
            # accumulating drift (keeps identity longer at the tail).
            boost = self.fg_boost * (1.0 + self.fg_boost_ramp * start_frame)
            boost = min(boost, self.fg_boost_max) if self.fg_boost_max > 0 else boost
            # frame-scheduled qsupp decay: ease bg-query suppression late so the
            # background re-grounds to the frame-0 anchor appearance.
            qsupp = self.fg_qsupp
            if self.fg_qsupp_decay > 0.0:
                qsupp = max(self.fg_qsupp_floor,
                            self.fg_qsupp * (1.0 - self.fg_qsupp_decay * start_frame))
            if self.fg_soft:
                return _fgsink_soft(q, k, v, c0, self.grid_rows, self.grid_cols,
                                    self.fg_box, boost, self.fg_sigma,
                                    qsupp, c1, self.fg_vhold,
                                    self.fg_vhold_adain)
            return _fgsink_attention(q, k, v, c0, self.grid_rows,
                                     self.grid_cols, self.fg_box, self.fg_boost)

        if self.mode in ("anchor_hold", "adain_hold"):
            within0 = max(c0, c1)  # within-frame columns start
            if not has_cross:
                # first block: capture the within-frame (robot anchor) values
                self._anchor_v[layer_idx] = v[:, within0:].detach()
                return None  # normal SDPA
            av = self._anchor_v.get(layer_idx)
            lamv = self.lambda_v * self._gate(layer_idx)
            if av is None or lamv <= 0.0:
                return None
            if self.mode == "adain_hold":
                return _adain_attention(q, k, v, c1, lk, av, lamv)
            return _vhold_attention(q, k, v, c1, lk, av, lamv)

        if self.mode == "record":
            # Stash source q + all post-sink keys (cross+within), in call order.
            self._buffer.append((q.detach(), k[:, c0:].detach()))
            return None  # source uses ordinary SDPA

        # ---- replay ----------------------------------------------------
        entry = self._buffer[self._ptr] if self._ptr < len(self._buffer) else None
        self._ptr += 1
        gate = self._gate(layer_idx)
        lam = self.lambda_max * gate      # cross-frame (camera) routing
        lamw = self.lambda_within * gate  # within-frame (pose) structure
        if entry is None or (lam <= 0.0 and lamw <= 0.0):
            return None  # no injection -> ordinary target SDPA

        q_s, k_s = entry  # k_s columns correspond to target cols [c0:lk]
        return _blended_attention(q, k, v, q_s, k_s, c0, c1, lk, lam, lamw)


def _blended_attention(q, k, v, q_s, k_s, c0, c1, lk, lam, lamw):
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    qh = q.permute(0, 2, 1, 3).float()          # [B,H,Lq,D]
    kh = k.permute(0, 2, 1, 3).float()          # [B,H,Lk,D]
    vh = v.permute(0, 2, 1, 3)                   # [B,H,Lk,D]
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [B,H,Lq,Lk]

    qs = q_s.permute(0, 2, 1, 3).float()        # [B,H,Lq',D]
    ks = k_s.permute(0, 2, 1, 3).float()        # [B,H,Z,D]  source cols [c0:..]
    z_s = torch.matmul(qs, ks.transpose(-1, -2)) * scale     # [B,H,Lq',Z]
    Z = z_s.shape[-1]
    nrow = min(logits.shape[2], z_s.shape[2])   # align query rows defensively

    c0 = max(0, min(c0, lk))
    c1 = max(c0, min(c1, lk))
    ncross = c1 - c0                            # cross cols in target
    # cross-frame region: target [c0:c1] <-> z_s[:, :ncross]   (camera motion)
    if lam > 0.0 and ncross > 0:
        w = min(ncross, Z)
        if w > 0:
            logits[:, :, :nrow, c0:c0 + w] = (
                (1.0 - lam) * logits[:, :, :nrow, c0:c0 + w]
                + lam * z_s[:, :, :nrow, :w])
    # within-frame region: target [c1:lk] <-> z_s[:, ncross:]  (pose/structure)
    nwithin = lk - c1
    if lamw > 0.0 and nwithin > 0 and Z - ncross > 0:
        w = min(nwithin, Z - ncross)
        if w > 0:
            logits[:, :, :nrow, c1:c1 + w] = (
                (1.0 - lamw) * logits[:, :, :nrow, c1:c1 + w]
                + lamw * z_s[:, :, :nrow, ncross:ncross + w])
    attn_w = torch.softmax(logits, dim=-1).to(vh.dtype)
    x = torch.matmul(attn_w, vh)                 # [B,H,Lq,D]
    return x.permute(0, 2, 1, 3)                 # [B,Lq,H,D]


def _vhold_attention(q, k, v, c1, lk, av, lamv):
    """Keep target structure (Q,K logits) but blend within-frame VALUES toward
    the anchor's values -> appearance held to anchor, motion left free."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).clone()           # [B,H,Lk,D]
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    avh = av.permute(0, 2, 1, 3)                  # [B,H,nw0,D]
    w = min(lk - c1, avh.shape[2])
    if w > 0:
        vh[:, :, c1:c1 + w, :] = ((1.0 - lamv) * vh[:, :, c1:c1 + w, :]
                                  + lamv * avh[:, :, :w, :].to(vh.dtype))
    attn_w = torch.softmax(logits, dim=-1).to(vh.dtype)
    x = torch.matmul(attn_w, vh)
    return x.permute(0, 2, 1, 3)


def _adain_attention(q, k, v, c1, lk, av, lamv):
    """Keep structure (Q,K) and within-frame token-wise variation (articulation),
    but AdaIN the within-frame VALUES' per-channel stats to the anchor (identity).
    av: [B, nw0, H, D] anchor within-V. Stats over the token axis."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).clone()           # [B,H,Lk,D]
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale

    nw = lk - c1
    if nw > 0 and av.shape[1] > 0:
        cur = vh[:, :, c1:lk, :].float()          # [B,H,nw,D]
        avh = av.permute(0, 2, 1, 3).float()      # [B,H,nw0,D]
        eps = 1e-5
        mu_c = cur.mean(dim=2, keepdim=True)
        sd_c = cur.std(dim=2, keepdim=True) + eps
        mu_a = avh.mean(dim=2, keepdim=True)
        sd_a = avh.std(dim=2, keepdim=True) + eps
        adain = sd_a * (cur - mu_c) / sd_c + mu_a  # robot stats, Link articulation
        blended = (1.0 - lamv) * cur + lamv * adain
        vh[:, :, c1:lk, :] = blended.to(vh.dtype)
    attn_w = torch.softmax(logits, dim=-1).to(vh.dtype)
    x = torch.matmul(attn_w, vh)
    return x.permute(0, 2, 1, 3)


def _fgsink_attention(q, k, v, sink_tokens, rows, cols, fg_box, fg_boost=0.0):
    """SDPA but mask the PERIPHERAL columns of the sink frame(s): the current
    frame may attend the sink's center (subject) for identity, but not the
    sink's periphery (background) -> background/viewpoint not tethered."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [B,H,Lq,Lk]

    st = int(sink_tokens)
    fpt = rows * cols
    if st >= fpt and st % fpt == 0:
        r0, r1, c0, c1 = fg_box
        idx = torch.arange(fpt, device=logits.device)
        rr = idx // cols
        cc = idx % cols
        keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)  # [fpt]
        nf = st // fpt
        periph = (~keep).repeat(nf)                              # [st]
        col_mask = torch.zeros(logits.shape[-1], dtype=torch.bool,
                               device=logits.device)
        col_mask[:st] = periph
        logits = logits.masked_fill(col_mask.view(1, 1, 1, -1), float("-inf"))
        if fg_boost > 0.0:
            bias = torch.zeros(logits.shape[-1], device=logits.device,
                               dtype=logits.dtype)
            bias[:st] = keep.repeat(nf).to(logits.dtype) * fg_boost
            logits = logits + bias.view(1, 1, 1, -1)
        # query-side: only foreground (center) current tokens may attend the
        # sink; background queries see only recent frames -> free to travel.
        lq = logits.shape[2]
        if lq % fpt == 0:
            nfq = lq // fpt
            qbg = (~keep).repeat(nfq)                # [Lq] background queries
            sinkcols = torch.zeros(logits.shape[-1], dtype=torch.bool,
                                   device=logits.device)
            sinkcols[:st] = True
            full = qbg.view(1, 1, lq, 1) & sinkcols.view(1, 1, 1, -1)
            logits = logits.masked_fill(full, float("-inf"))
    attn_w = torch.softmax(logits, dim=-1).to(vh.dtype)
    x = torch.matmul(attn_w, vh)
    return x.permute(0, 2, 1, 3)


def _fgsink_soft(q, k, v, sink_tokens, rows, cols, fg_box, fg_boost, sigma,
                 qsupp=0.0, c1=None, vhold=0.0, vhold_adain=False):
    """Feathered (no hard seam) version: bias sink columns by a smooth gaussian
    centered on the subject — center cols softly boosted, periphery softly
    suppressed, zero hard masking, no query mask. Gentle -> preserves quality.

    Optional subject-region VALUE anchor (vhold>0): blend the current within-frame
    values [c1:lk] toward the frame-0 anchor values [0:fpt], ONLY inside fg_box, to
    pin subject appearance/identity over a long rollout (and pull the spawned glider
    region back to the glider-free anchor) without freezing routing/articulation."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    st0 = int(sink_tokens)
    fpt = rows * cols
    # subject-region value anchor (before attention) -------------------------
    if (vhold > 0.0 and c1 is not None and st0 >= fpt and st0 % fpt == 0):
        c1 = int(c1)
        nw = v.shape[1] - c1
        if nw > 0 and nw % fpt == 0:
            r0, r1, cc0, cc1 = fg_box
            idx = torch.arange(fpt, device=v.device)
            rr = idx // cols; ccc = idx % cols
            keep = ((rr >= r0) & (rr < r1) & (cc0 <= ccc) & (ccc < cc1))  # [fpt]
            v = v.clone()
            v0 = v[:, :fpt]                                # frame-0 anchor [B,fpt,H,D]
            km = keep.view(1, fpt, 1, 1)
            nfw = nw // fpt
            for j in range(nfw):
                s = c1 + j * fpt
                blk = v[:, s:s + fpt]                      # [B,fpt,H,D]
                if vhold_adain:
                    cur = blk.float(); anc = v0.float()
                    m = keep.view(1, fpt, 1, 1)
                    eps = 1e-5
                    # per-(channel) stats over subject tokens only
                    cm = (cur * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
                    am = (anc * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
                    cs = (((cur - cm) ** 2 * m).sum(1, keepdim=True)
                          / m.sum(1, keepdim=True)).sqrt() + eps
                    as_ = (((anc - am) ** 2 * m).sum(1, keepdim=True)
                           / m.sum(1, keepdim=True)).sqrt() + eps
                    adain = as_ * (cur - cm) / cs + am
                    new = ((1.0 - vhold) * cur + vhold * adain).to(blk.dtype)
                else:
                    new = (1.0 - vhold) * blk + vhold * v0
                v[:, s:s + fpt] = torch.where(km, new, blk)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    st = int(sink_tokens)
    fpt = rows * cols
    if st >= fpt and st % fpt == 0 and fg_boost != 0.0:
        r0, r1, c0, c1 = fg_box
        rc = 0.5 * (r0 + r1); cc = 0.5 * (c0 + c1)
        sr, sc = sigma
        idx = torch.arange(fpt, device=logits.device, dtype=torch.float32)
        rr = idx // cols; ccol = idx % cols
        w = torch.exp(-(((rr - rc) ** 2) / (2 * sr * sr)
                        + ((ccol - cc) ** 2) / (2 * sc * sc)))  # [fpt] in (0,1]
        # zero-mean feather: center>0 (boost), periphery<0 (suppress), smooth
        wb = (w - w.mean()) * fg_boost
        nf = st // fpt
        bias = torch.zeros(logits.shape[-1], device=logits.device,
                           dtype=logits.dtype)
        bias[:st] = wb.repeat(nf).to(logits.dtype)
        logits = logits + bias.view(1, 1, 1, -1)
        # soft query suppression: background current-frame queries (low w) softly
        # down-weight the sink so they ignore the anchor -> background travels.
        lq = logits.shape[2]
        if qsupp > 0.0 and lq % fpt == 0:
            nfq = lq // fpt
            wq = w.repeat(nfq)                          # [Lq] in (0,1]
            qbias = -(1.0 - wq) * qsupp                 # bg queries: negative
            sinkcol = torch.zeros(logits.shape[-1], device=logits.device,
                                  dtype=logits.dtype)
            sinkcol[:st] = 1.0
            logits = logits + (qbias.view(1, 1, lq, 1)
                               * sinkcol.view(1, 1, 1, -1)).to(logits.dtype)
    attn_w = torch.softmax(logits, dim=-1).to(vh.dtype)
    x = torch.matmul(attn_w, vh)
    return x.permute(0, 2, 1, 3)


# Module-level singleton used by the attention forward.
INJECTOR = _Injector()


def injected_sdpa(attn, q, k, v, sink_tokens, num_new_tokens, start_frame=0):
    """Drop-in for the cache-attention SDPA call in causal_model.py."""
    out = INJECTOR.maybe_apply(attn, q, k, v, sink_tokens, num_new_tokens,
                               start_frame)
    if out is not None:
        return out
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)
