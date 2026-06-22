"""OOD identity-preserving rollout (TRAINING-FREE).

Cracks the OOD-humanoid case: an off-distribution subject (robot, wukong, etc.)
keeps its identity AND articulates AND travels through a coherent world, where a
plain rollout would drift to the model's training prior (Link).

Method = Foreground-Anchored Attention (all in attn_injection.py, inference-only):
  - soft feathered fg_sink (no hard seam): anchor the subject region of frame 0
  - fg_boost: pin the subject to its own identity (beats the Link prior)
  - soft fg_qsupp: background queries down-weight the anchor -> world free to move
    and the subject free to articulate
  - mid-layer gating (0.2-0.8): don't distort detail layers -> coherent background
  - sink_size=1, local_attn_size=6
CRITICAL: fg_box must cover the ACTUAL subject (it's usually bottom-/mid-center in
third-person). Wrong box (e.g. on the sky) = no identity hold.

Usage:
  PYTHONPATH=/home/hal-kaiqin/FastVideo_attninj \
  /home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin/python run_ood_identity.py \
    --image <anchor.jpg> --box r0 r1 c0 c1   (grid 30x52 for 480x832)
"""
import argparse, time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT_DIR = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
            "WanGame2.1_1.3b_i2v/actions_801")
H, W = 480, 832


def w_setup(worker, box, boost, qsupp, sigma):
    from fastvideo.models.dits.matrixgame2.causal_model import (
        CausalMatrixGame2SelfAttention)
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = 1
            m.local_attn_size = 6
    INJECTOR.prepare(tf)
    INJECTOR.grid_rows, INJECTOR.grid_cols = 30, 52
    INJECTOR.fg_box = tuple(box)
    INJECTOR.fg_soft = True
    INJECTOR.fg_sigma = tuple(sigma)
    INJECTOR.start_fg_sink()
    INJECTOR.fg_boost = boost
    INJECTOR.fg_qsupp = qsupp
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    return (box, boost, qsupp)


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", default="W")
    ap.add_argument("--box", type=int, nargs=4, default=[14, 30, 21, 31],
                    help="r0 r1 c0 c1 of the SUBJECT in the 30x52 token grid")
    ap.add_argument("--boost", type=float, default=4.0)
    ap.add_argument("--qsupp", type=float, default=6.0)
    ap.add_argument("--sigma", type=float, nargs=2, default=[5.0, 5.0])
    ap.add_argument("--frames", type=int, default=117)
    ap.add_argument("--out", default="attn_injection_out/ood_identity")
    args = ap.parse_args()

    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print("[cfg]", rpc(gen, w_setup, args.box, args.boost, args.qsupp, args.sigma)[0],
          flush=True)

    nf = args.frames
    raw = np.load(f"{_ACT_DIR}/{args.action}.npy", allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:nf, :4])
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:nf, :])
    grid = torch.tensor([(nf + 3) // 4, H // 8, W // 8])
    t0 = time.time()
    gen.generate_video(
        prompt="", image_path=args.image, mouse_cond=mo.unsqueeze(0),
        keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=nf,
        height=H, width=W, num_inference_steps=4, seed=42,
        output_path=args.out, save_video=True)
    print(f"[done] {time.time()-t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
