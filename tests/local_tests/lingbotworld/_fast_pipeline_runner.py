# SPDX-License-Identifier: Apache-2.0
"""Standalone pipeline-stage parity runner for LingBot-World-Fast.

Run as a subprocess by ``test_lingbotworld_fast_pipeline_parity.py``. Drives one
deterministic DMD step (chunk 0) through the real
``LingBotWorldFastCausalDenoisingStage`` and through the official ``WanModelFast``
on identical inputs, and compares the resulting flow-x0 latent. This validates
the stage's pipeline-specific work -- the Wan-2.1 36-channel image concat and the
flow->x0 conversion -- against the official, deterministically (no re-noise
feedback).

The full multi-step AR loop is intentionally NOT the parity target: the ~4%
per-forward cross-implementation difference (component cosine 0.999) amplifies
through the iterative denoise + re-noise loop, so full-trajectory pixel parity is
ill-posed (a known property of iterative samplers). Multi-step orchestration is
covered by the pipeline smoke (coherent generation) and by the chunk-1 KV-cache
component parity.

The official runs first (before any FastVideo model is built) and both sides use
the MATH SDPA backend -- the flash / mem-efficient backends are numerically
unstable on GB200/Blackwell for these shapes. Exits 0 iff the comparison passes.

Usage: python _fast_pipeline_runner.py
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.local_tests.lingbotworld._fast_parity_runner import (
    DTYPE, FSL, KV_SIZE, LH, LW, NF, REF_REPO, _init_self_kv, _load_sd,
    _resolve_weights, _sdpa)

SEED_INPUT = 99
COS_MIN, REL_MAX = 0.99, 0.06
TIMESTEPS_INDEX = [0, 179, 358, 679]


def _official_cross():
    return [{"k": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
             "v": torch.zeros(1, 512, 40, 128, dtype=DTYPE, device="cuda"),
             "is_init": torch.tensor(0, dtype=torch.int32, device="cuda")} for _ in range(40)]


def _build_official(sd):
    if str(REF_REPO) not in sys.path:
        sys.path.insert(0, str(REF_REPO))
    import wan.modules.attention as watt
    import wan.modules.model_fast as wmf
    watt.attention = watt.flash_attention = wmf.attention = wmf.flash_attention = _sdpa
    from wan.modules.model_fast import WanModelFast
    off = WanModelFast(model_type="t2v", control_type="cam", patch_size=(1, 2, 2), text_len=512,
                       in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
                       num_heads=40, num_layers=40, local_attn_size=-1, sink_size=9, qk_norm=True,
                       cross_attn_norm=True, eps=1e-6)
    miss, unexp = off.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    return off.to("cuda", DTYPE).eval().requires_grad_(False)


def _official_x0(off, x16, y20, ctx, t0, scheduler):
    from fastvideo.models.utils import pred_noise_to_pred_video
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE), sdpa_kernel([SDPBackend.MATH]):
        noise_in = x16.permute(0, 2, 1, 3, 4).clone()
        pred = off(x=[x16[0]], t=torch.tensor([float(t0)], device="cuda"), context=[ctx[0]],
                   seq_len=NF * FSL, y=[y20[0]], dit_cond_dict=None, kv_cache=_init_self_kv(),
                   crossattn_cache=_official_cross(), current_start=0, max_attention_size=KV_SIZE,
                   frame_seqlen=FSL)[0].unsqueeze(0).permute(0, 2, 1, 3, 4)
        x0 = pred_noise_to_pred_video(pred.flatten(0, 1), noise_in.flatten(0, 1),
                                      t0.repeat(NF), scheduler).unflatten(0, pred.shape[:2])
    return x0.permute(0, 2, 1, 3, 4)   # [1, 16, NF, LH, LW]


def _build_fastvideo(sd):
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    from fastvideo.configs.models.dits.lingbotworld import CausalLingBotWorldVideoConfig
    from fastvideo.models.dits.lingbotworld.causal_model import CausalLingBotWorldTransformer3DModel
    cfg = CausalLingBotWorldVideoConfig()
    with torch.device("cuda"):
        fv = CausalLingBotWorldTransformer3DModel(cfg, hf_config={})
    mapping = cfg.arch_config.param_names_mapping

    def amap(k):
        for p, r in mapping.items():
            k = re.sub(p, r, k)
        return k
    miss, unexp = fv.load_state_dict({amap(k): v for k, v in sd.items()}, strict=False)
    assert not miss and not unexp
    return fv.to("cuda", DTYPE).eval().requires_grad_(False)


def _fv_stage_x0(fv, x16, y20, ctx, t0, scheduler):
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.stages.lingbotworld_fast_denoising import (
        LingBotWorldFastCausalDenoisingStage)
    stage = LingBotWorldFastCausalDenoisingStage(transformer=fv, scheduler=scheduler,
                                                 pipeline=None, vae=None)
    stage.frame_seq_length = FSL
    batch = SimpleNamespace(generator=torch.Generator(device="cpu").manual_seed(0))
    kv = stage._initialize_kv_cache(1, NF, DTYPE, x16.device)
    cr = stage._initialize_crossattn_cache(1, fv.text_len, DTYPE, x16.device)
    ts = torch.tensor([float(t0)], device="cuda")  # single deterministic DMD step
    with sdpa_kernel([SDPBackend.MATH]), set_forward_context(current_timestep=0, attn_metadata=None):
        x0 = stage._process_single_block(x16, [ctx], y20, None, 0, NF, ts, kv, cr, batch,
                                         DTYPE, True, None)
    return x0   # [1, 16, NF, LH, LW]


def main():
    sd = _load_sd(_resolve_weights())
    g = torch.Generator(device="cuda").manual_seed(SEED_INPUT)
    x16 = torch.randn(1, 16, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)
    y20 = torch.randn(1, 20, NF, LH, LW, generator=g, device="cuda", dtype=DTYPE)
    ctx = torch.randn(1, 512, 4096, generator=g, device="cuda", dtype=DTYPE)

    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler)
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sched.set_timesteps(num_inference_steps=1000, device="cuda", shift=3.0)
    t0 = sched.timesteps[TIMESTEPS_INDEX[0]].to("cuda")

    off = _build_official(sd)                       # built + run before any FastVideo model
    x0_off = _official_x0(off, x16, y20, ctx, t0, sched)
    del off
    torch.cuda.empty_cache()

    fv = _build_fastvideo(sd)
    x0_fv = _fv_stage_x0(fv, x16, y20, ctx, t0, sched)

    a, b = x0_fv.float().flatten(), x0_off.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-8)
    passed = cos > COS_MIN and rel < REL_MAX
    print(f"RESULT pipeline cosine={cos:.6f} relMAE={rel:.4f} "
          f"fv_absmean={a.abs().mean():.4e} ref_absmean={b.abs().mean():.4e} "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    print("OVERALL", "PASS" if passed else "FAIL", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
