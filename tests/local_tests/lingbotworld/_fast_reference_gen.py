# SPDX-License-Identifier: Apache-2.0
"""Standalone official-reference generator for the LingBot-World-Fast parity test.

Run as a subprocess by ``test_lingbotworld_fast_parity.py``. Imports ONLY torch +
the official ``wan`` repo (no pytest, no fastvideo): the official ``WanModelFast``
is numerically unstable on GB200/Blackwell in some process configurations, so it
is exercised here in a minimal clean process. The official attention is routed
through the MATH SDPA backend (flash/mem-efficient are unstable on Blackwell for
these shapes). Saves the matched inputs and the official denoised-noise outputs.

Usage: python _fast_reference_gen.py <out_path>
"""
import glob
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
REF_REPO = Path(os.getenv("LINGBOT_WORLD_REPO", REPO_ROOT / "reference" / "lingbot-world"))

LH, LW, NF = 60, 104, 3
FSL = LH * LW // 4
KV_SIZE = 21 * FSL
DTYPE = torch.bfloat16


def resolve_weights():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download("robbyant/lingbot-world-fast-diffusers")
    except Exception:
        return None


def load_sd(snap):
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "transformer", "*.safetensors"))):
        sd.update(load_file(f))
    return sd


def force_official_sdpa():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import wan.modules.attention as watt
    import wan.modules.model_fast as wmf

    def sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
             q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
             dtype=torch.bfloat16, **kw):
        if q_scale is not None:
            q = q * q_scale
        with sdpa_kernel([SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).to(dtype), k.transpose(1, 2).to(dtype),
                v.transpose(1, 2).to(dtype), is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2).contiguous().type(q.dtype)

    watt.attention = sdpa
    watt.flash_attention = sdpa
    wmf.attention = sdpa
    wmf.flash_attention = sdpa


def build_official(sd):
    if str(REF_REPO) not in sys.path:
        sys.path.insert(0, str(REF_REPO))
    import wan.modules.model_fast  # noqa: F401
    force_official_sdpa()
    from wan.modules.model_fast import WanModelFast
    off = WanModelFast(model_type="t2v", control_type="cam", patch_size=(1, 2, 2),
                       text_len=512, in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256,
                       text_dim=4096, out_dim=16, num_heads=40, num_layers=40,
                       local_attn_size=-1, sink_size=9, qk_norm=True,
                       cross_attn_norm=True, eps=1e-6)
    miss, unexp = off.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"official strict-load: missing={miss[:3]} unexpected={unexp[:3]}"
    return off.to("cuda", DTYPE).eval().requires_grad_(False)


def init_self_kv(n=40):
    return [{"k": torch.zeros(1, KV_SIZE, 40, 128, dtype=DTYPE, device="cuda"),
             "v": torch.zeros(1, KV_SIZE, 40, 128, dtype=DTYPE, device="cuda"),
             "global_end_index": torch.tensor([0], dtype=torch.long, device="cuda"),
             "local_end_index": torch.tensor([0], dtype=torch.long, device="cuda")} for _ in range(n)]


def cross(n=40):
    return [{"k": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
             "v": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
             "is_init": torch.tensor(0, dtype=torch.int32, device="cuda")} for _ in range(n)]


def official_forward(off, x16, y20, ctx, t_val, kv, cr, current_start):
    return off(x=[x16[0]], t=torch.tensor([t_val], device="cuda"), context=[ctx[0]],
               seq_len=NF * FSL, y=[y20[0]], dit_cond_dict=None, kv_cache=kv,
               crossattn_cache=cr, current_start=current_start,
               max_attention_size=KV_SIZE, frame_seqlen=FSL)[0]


def generate_reference(out_path):
    snap = resolve_weights()
    if snap is None or not REF_REPO.exists():
        torch.save({"available": False}, out_path)
        return
    off = build_official(load_sd(snap))

    g = torch.Generator(device="cuda").manual_seed(1234)
    x16 = torch.randn(1, 16, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)
    y20 = torch.randn(1, 20, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)
    ctx = torch.randn(1, 512, 4096, generator=g, device="cuda", dtype=DTYPE)

    g2 = torch.Generator(device="cuda").manual_seed(7)
    cx0 = torch.randn(1, 16, NF, LH, LW, generator=g2, device="cuda", dtype=DTYPE)
    cy0 = torch.randn(1, 20, NF, LH, LW, generator=g2, device="cuda", dtype=DTYPE)
    cx1 = torch.randn(1, 16, NF, LH, LW, generator=g2, device="cuda", dtype=DTYPE)
    cy1 = torch.zeros(1, 20, NF, LH, LW, device="cuda", dtype=DTYPE)
    cctx = torch.randn(1, 512, 4096, generator=g2, device="cuda", dtype=DTYPE)

    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out0 = official_forward(off, x16, y20, ctx, 500.0, init_self_kv(), cross(), 0)
        okv, ocr = init_self_kv(), cross()
        official_forward(off, cx0, cy0, cctx, 0.0, okv, ocr, 0)
        out1 = official_forward(off, cx1, cy1, cctx, 500.0, okv, ocr, NF * FSL)

    torch.save({"available": True,
                "x16": x16.cpu(), "y20": y20.cpu(), "ctx": ctx.cpu(), "out0": out0.cpu(),
                "cx0": cx0.cpu(), "cy0": cy0.cpu(), "cx1": cx1.cpu(), "cy1": cy1.cpu(),
                "cctx": cctx.cpu(), "out1": out1.cpu()}, out_path)


if __name__ == "__main__":
    generate_reference(sys.argv[1])
