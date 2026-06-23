# SPDX-License-Identifier: Apache-2.0
"""Standalone component-parity runner for LingBot-World-Act.

Run as a subprocess by ``test_lingbotworld_act_parity.py``. Loads the official
base ``WanModel(control_type='act')`` and the FastVideo
``LingBotWorldTransformer3DModel`` (Act config, control_dim=7) from the SAME
converted high-noise expert weights, feeds identical inputs (36-ch latent, text,
timestep, and a 7x64 act2cam Plucker tensor), and compares the full-sequence
denoised-noise output. This validates the Act DiT port + the act2cam control
width (control_dim=7) against the official model.

Both models use the MATH SDPA backend (flash / mem-efficient are unstable on
GB200/Blackwell for these shapes); the official model is built and run first.
Exits 0 iff the comparison passes the bounds.

Usage: python _act_parity_runner.py <converted_act_diffusers_dir>
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
CONTROL_DIM = 7
DTYPE = torch.bfloat16
COS_MIN, REL_MAX, DRIFT_MAX = 0.99, 0.06, 0.05


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


def _load_sd(transformer_dir):
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


def _official_output(sd, x16, y20, ctx, c2ws, t_val):
    if str(REF_REPO) not in sys.path:
        sys.path.insert(0, str(REF_REPO))
    import wan.modules.attention as watt
    import wan.modules.model as wm
    watt.attention = watt.flash_attention = wm.attention = wm.flash_attention = _sdpa
    from wan.modules.model import WanModel
    off = WanModel(model_type="i2v", control_type="act", patch_size=(1, 2, 2), text_len=512,
                   in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
                   num_heads=40, num_layers=40, eps=1e-6)
    miss, unexp = off.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"official load: missing={miss[:3]} unexpected={unexp[:3]}"
    off = off.to("cuda", DTYPE).eval().requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE), sdpa_kernel([SDPBackend.MATH]):
        out = off(x=[x16[0]], t=torch.tensor([t_val], device="cuda"), context=[ctx[0]],
                  seq_len=NF * FSL, y=[y20[0]],
                  dit_cond_dict={"c2ws_plucker_emb": (c2ws,)})[0]
    return out   # [16, NF, LH, LW]


def _fastvideo_output(sd, x16, y20, ctx, c2ws, t_val):
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    from fastvideo.configs.models.dits.lingbotworld import LingBotWorldActVideoConfig
    from fastvideo.models.dits.lingbotworld.model import LingBotWorldTransformer3DModel
    from fastvideo.forward_context import set_forward_context
    cfg = LingBotWorldActVideoConfig()
    with torch.device("cuda"):
        fv = LingBotWorldTransformer3DModel(cfg, hf_config={})
    mapping = cfg.arch_config.param_names_mapping

    def amap(k):
        for p, r in mapping.items():
            k = re.sub(p, r, k)
        return k
    miss, unexp = fv.load_state_dict({amap(k): v for k, v in sd.items()}, strict=False)
    assert not miss and not unexp, f"fastvideo load: missing={miss[:3]} unexpected={unexp[:3]}"
    fv = fv.to("cuda", DTYPE).eval().requires_grad_(False)
    t_fv = torch.tensor([t_val], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE), sdpa_kernel([SDPBackend.MATH]), \
         set_forward_context(current_timestep=0, attn_metadata=None):
        out = fv(torch.cat([x16, y20], dim=1), [ctx], t_fv, c2ws_plucker_emb=c2ws)[0]
    return out   # [16, NF, LH, LW]


def main():
    transformer_dir = os.path.join(sys.argv[1], "transformer")
    sd = _load_sd(transformer_dir)
    g = torch.Generator(device="cuda").manual_seed(2024)
    x16 = torch.randn(1, 16, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)
    y20 = torch.randn(1, 20, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)
    ctx = torch.randn(1, 512, 4096, generator=g, device="cuda", dtype=DTYPE)
    c2ws = torch.randn(1, CONTROL_DIM * 64, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)

    off = _official_output(sd, x16, y20, ctx, c2ws, 500.0)   # official first
    fv = _fastvideo_output(sd, x16, y20, ctx, c2ws, 500.0)

    a, b = fv.float().flatten(), off.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-8)
    drift = abs(a.abs().mean().item() - b.abs().mean().item()) / (b.abs().mean().item() + 1e-8)
    passed = cos > COS_MIN and rel < REL_MAX and drift < DRIFT_MAX
    print(f"RESULT act_transformer cosine={cos:.6f} relMAE={rel:.4f} drift={drift:.4f} "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    print("OVERALL", "PASS" if passed else "FAIL", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
