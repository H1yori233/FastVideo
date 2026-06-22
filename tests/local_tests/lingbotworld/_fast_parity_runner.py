# SPDX-License-Identifier: Apache-2.0
"""Standalone component-parity runner for LingBot-World-Fast.

Run as a subprocess by ``test_lingbotworld_fast_parity.py``. Loads the official
``WanModelFast`` and the FastVideo ``CausalLingBotWorldTransformer3DModel`` from
the same ``robbyant/lingbot-world-fast-diffusers`` weights, feeds identical
inputs through a per-layer KV cache, and compares the denoised-noise outputs for
chunk 0 (forward) and chunk 1 (block-causal KV-cache reuse).

Both models use the MATH SDPA backend: flash_attn / the flash & mem-efficient
SDPA backends are numerically unstable on GB200/Blackwell for these shapes (the
official's bf16 output flips to ~2.5x-off garbage), while MATH is deterministic.
The official is built and run before the FastVideo model to keep its numerics
clean. Prints ``RESULT <name> cosine=.. relMAE=.. drift=..`` lines and exits 0
iff every comparison passes the bounds.

Usage: python _fast_parity_runner.py
"""
import glob
import os
import re
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[3]
REF_REPO = Path(os.getenv("LINGBOT_WORLD_REPO", REPO_ROOT / "reference" / "lingbot-world"))
LH, LW, NF = 60, 104, 3
FSL = LH * LW // 4
KV_SIZE = 21 * FSL
DTYPE = torch.bfloat16
COS_MIN, REL_MAX, DRIFT_MAX = 0.99, 0.06, 0.05


def _resolve_weights():
    from huggingface_hub import snapshot_download
    return snapshot_download("robbyant/lingbot-world-fast-diffusers")


def _load_sd(snap):
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "transformer", "*.safetensors"))):
        sd.update(load_file(f))
    return sd


def _sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
          q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
          dtype=torch.bfloat16, **kw):
    if q_scale is not None:
        q = q * q_scale
    with sdpa_kernel([SDPBackend.MATH]):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).to(dtype), k.transpose(1, 2).to(dtype),
            v.transpose(1, 2).to(dtype), is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().type(q.dtype)


def _init_self_kv(n=40):
    return [{"k": torch.zeros(1, KV_SIZE, 40, 128, dtype=DTYPE, device="cuda"),
             "v": torch.zeros(1, KV_SIZE, 40, 128, dtype=DTYPE, device="cuda"),
             "global_end_index": torch.tensor([0], dtype=torch.long, device="cuda"),
             "local_end_index": torch.tensor([0], dtype=torch.long, device="cuda")} for _ in range(n)]


def _inputs():
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
    return x16, y20, ctx, cx0, cy0, cx1, cy1, cctx


def _official_outputs(sd, inp):
    if str(REF_REPO) not in sys.path:
        sys.path.insert(0, str(REF_REPO))
    import wan.modules.model_fast as wmf
    import wan.modules.attention as watt
    watt.attention = watt.flash_attention = wmf.attention = wmf.flash_attention = _sdpa
    from wan.modules.model_fast import WanModelFast
    off = WanModelFast(model_type="t2v", control_type="cam", patch_size=(1, 2, 2), text_len=512,
                       in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
                       num_heads=40, num_layers=40, local_attn_size=-1, sink_size=9, qk_norm=True,
                       cross_attn_norm=True, eps=1e-6)
    miss, unexp = off.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"official load: {miss[:3]} {unexp[:3]}"
    off = off.to("cuda", DTYPE).eval().requires_grad_(False)
    x16, y20, ctx, cx0, cy0, cx1, cy1, cctx = inp

    def fwd(x, y, c, t, kv, cr, cs):
        return off(x=[x[0]], t=torch.tensor([t], device="cuda"), context=[c[0]], seq_len=NF * FSL,
                   y=[y[0]], dit_cond_dict=None, kv_cache=kv, crossattn_cache=cr,
                   current_start=cs, max_attention_size=KV_SIZE, frame_seqlen=FSL)[0]

    def cr():
        return [{"k": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
                 "v": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
                 "is_init": torch.tensor(0, dtype=torch.int32, device="cuda")} for _ in range(40)]
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out0 = fwd(x16, y20, ctx, 500.0, _init_self_kv(), cr(), 0)
        okv, ocr = _init_self_kv(), cr()
        fwd(cx0, cy0, cctx, 0.0, okv, ocr, 0)
        out1 = fwd(cx1, cy1, cctx, 500.0, okv, ocr, NF * FSL)
    return out0, out1


def _fastvideo_outputs(sd, inp):
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    from fastvideo.configs.models.dits.lingbotworld import CausalLingBotWorldVideoConfig
    from fastvideo.models.dits.lingbotworld.causal_model import CausalLingBotWorldTransformer3DModel
    from fastvideo.forward_context import set_forward_context
    cfg = CausalLingBotWorldVideoConfig()
    with torch.device("cuda"):
        fv = CausalLingBotWorldTransformer3DModel(cfg, hf_config={})
    mapping = cfg.arch_config.param_names_mapping

    def amap(k):
        for p, r in mapping.items():
            k = re.sub(p, r, k)
        return k
    miss, unexp = fv.load_state_dict({amap(k): v for k, v in sd.items()}, strict=False)
    assert not miss and not unexp, f"fastvideo load: {miss[:3]} {unexp[:3]}"
    fv = fv.to("cuda", DTYPE).eval().requires_grad_(False)
    x16, y20, ctx, cx0, cy0, cx1, cy1, cctx = inp

    def cr():
        return [{"k": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
                 "v": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
                 "is_init": False} for _ in range(40)]

    def fwd(x, y, c, t, kv, crc, cs, sf):
        tt = t * torch.ones(1, NF, device="cuda", dtype=torch.long)
        return fv(torch.cat([x, y], dim=1), [c], tt, kv_cache=kv, crossattn_cache=crc,
                  current_start=cs, start_frame=sf, encoder_hidden_states_image=[])[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE), sdpa_kernel([SDPBackend.MATH]), \
         set_forward_context(current_timestep=0, attn_metadata=None):
        out0 = fwd(x16, y20, ctx, 500.0, _init_self_kv(), cr(), 0, 0)
        fkv, fcr = _init_self_kv(), cr()
        fwd(cx0, cy0, cctx, 0.0, fkv, fcr, 0, 0)
        out1 = fwd(cx1, cy1, cctx, 500.0, fkv, fcr, NF * FSL, NF)
    return out0, out1


def main():
    snap = _resolve_weights()
    sd = _load_sd(snap)
    inp = _inputs()
    off0, off1 = _official_outputs(sd, inp)   # official first, before FastVideo runs
    fv0, fv1 = _fastvideo_outputs(sd, inp)
    ok = True
    for name, a, b in [("chunk0", fv0, off0), ("chunk1", fv1, off1)]:
        a, b = a.float().flatten(), b.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-8)
        drift = abs(a.abs().mean().item() - b.abs().mean().item()) / (b.abs().mean().item() + 1e-8)
        passed = cos > COS_MIN and rel < REL_MAX and drift < DRIFT_MAX
        ok = ok and passed
        print(f"RESULT {name} cosine={cos:.6f} relMAE={rel:.4f} drift={drift:.4f} "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
    print("OVERALL", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
