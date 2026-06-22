"""Apply cross-frame attention injection to Matrix-Game 2 causal inference.

Runs three rollouts on one loaded generator, all sharing actions + seed:
  1. source   (RECORD): rollout on SRC anchor, stashing source cross-frame q/k
  2. injected (REPLAY): rollout on TGT anchor, blending source routing in
  3. baseline (OFF)   : rollout on TGT anchor, no injection (for comparison)

Sanity mode: with SRC == TGT, (2) should match (3) almost exactly, proving the
record/replay harness + logit blend are non-destructive.

Run:
  PYTHONPATH=/home/hal-kaiqin/FastVideo_attninj \
  /home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin/python apply_attn_injection.py \
    --src assets/images/<a>.png --tgt assets/images/<b>.png
"""
import argparse
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch

from fastvideo import VideoGenerator

# Our Zelda world model (paper "Ours", 4-step DMD-distilled, 480x832).
# symlink -> mg_causal_zelda; name carries the "matrix-game-2.0" token the
# registry detector needs to resolve the causal-DMD config preset.
MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-ours"

# Matched triple from validation_physics.json (action W):
#   source = the in-dist Zelda anchor for action W (well-calibrated dynamics)
#   target = an OOD anchor (robot) driven by the SAME action
_FV = "/home/hal-kaiqin/FastVideo"
DEFAULT_SRC = (f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/"
               "zelda/5TTrlqAguhQ_chunk_0484/frames/run_ahead.png")
DEFAULT_TGT = f"{_FV}/assets/third-person/robot.png"
DEFAULT_ACTION = (f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/"
                  "actions_801/W.npy")
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out")

H, W = 480, 832
NUM_FRAMES = 57
NUM_STEPS = 4
SEED = 42


# ---- functions that execute inside the worker process --------------------
def w_prepare(worker):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    INJECTOR.prepare(tf)
    return INJECTOR.num_layers


def w_config(worker, lam, steps, lo, hi):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.lambda_max = lam
    INJECTOR.inject_steps = tuple(steps)
    INJECTOR.layer_lo = lo
    INJECTOR.layer_hi = hi
    return (INJECTOR.lambda_max, INJECTOR.inject_steps,
            INJECTOR.layer_lo, INJECTOR.layer_hi)


def w_mode(worker, mode):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    if mode == "record":
        INJECTOR.start_record()
    elif mode == "replay":
        INJECTOR.start_replay()
    else:
        INJECTOR.off()
    return INJECTOR.mode


def w_bufinfo(worker):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    return (INJECTOR.mode, len(INJECTOR._buffer), INJECTOR._ptr)


def rpc(gen, fn, *args):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=args)


def load_actions(action_path):
    """Load {keyboard:(T,6), mouse:(T,2)} npy; slice kb 6->4 (WASD), to NF."""
    raw = np.load(action_path, allow_pickle=True).item()
    kb = np.asarray(raw["keyboard"], dtype=np.float32)[:NUM_FRAMES, :4]
    mouse = np.asarray(raw["mouse"], dtype=np.float32)[:NUM_FRAMES, :]
    if kb.shape[0] < NUM_FRAMES:  # pad short clips with zeros
        kb = np.concatenate(
            [kb, np.zeros((NUM_FRAMES - kb.shape[0], 4), np.float32)])
        mouse = np.concatenate(
            [mouse, np.zeros((NUM_FRAMES - mouse.shape[0], 2), np.float32)])
    return torch.from_numpy(kb), torch.from_numpy(mouse)


def rollout(gen, image_path, action_path, out_dir):
    kb, mouse = load_actions(action_path)
    grid_sizes = torch.tensor([(NUM_FRAMES + 3) // 4, H // 8, W // 8])
    t0 = time.time()
    gen.generate_video(
        prompt="",
        image_path=image_path,
        mouse_cond=mouse.unsqueeze(0),
        keyboard_cond=kb.unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=NUM_FRAMES,
        height=H,
        width=W,
        num_inference_steps=NUM_STEPS,
        seed=SEED,
        output_path=str(out_dir),
        save_video=True,
    )
    print(f"[rollout] {out_dir.name}: {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--tgt", default=DEFAULT_TGT)
    ap.add_argument("--action", default=DEFAULT_ACTION)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[init] loading {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )

    nlayers = rpc(gen, w_prepare)[0]
    cfg = rpc(gen, w_config, args.lam, args.steps, args.lo, args.hi)[0]
    print(f"[init] self-attn layers={nlayers}  config={cfg}", flush=True)
    print(f"[init] src={Path(args.src).name}  tgt={Path(args.tgt).name}",
          flush=True)

    # 1) source rollout, recording cross-frame q/k
    rpc(gen, w_mode, "record")
    rollout(gen, args.src, args.action, out / "1_source_record")
    print(f"[buf] after record: {rpc(gen, w_bufinfo)[0]}", flush=True)

    # 2) target rollout, replaying (injected)
    rpc(gen, w_mode, "replay")
    rollout(gen, args.tgt, args.action, out / "2_target_injected")
    print(f"[buf] after replay: {rpc(gen, w_bufinfo)[0]}", flush=True)

    # 3) target baseline, no injection
    rpc(gen, w_mode, "off")
    rollout(gen, args.tgt, args.action, out / "3_target_baseline")

    print(f"[done] outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
