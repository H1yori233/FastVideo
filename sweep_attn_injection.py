"""Sweep cross-frame attention injection over targets / actions / lambda.

Loads the model once; for each group records the source rollout once, runs the
baseline once, then replays at each lambda (the recorded buffer is independent
of lambda).  Outputs go to attn_injection_out/<group>/{source,baseline,lamXX}.
"""
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch

from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-ours"
_FV = "/home/hal-kaiqin/FastVideo"
_ZELDA = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801"
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out")

H, W = 480, 832
NUM_FRAMES = 57
NUM_STEPS = 4
SEED = 42

SNOW = f"{_FV}/assets/images/mixkit-curve-on-a-snowy-forest-road-3317.png"
SKI = f"{_FV}/assets/images/mixkit-skiers-on-a-snowy-slope-3327.png"
ROBOT = f"{_FV}/assets/third-person/robot.png"

# matched Zelda source per action (well-calibrated dynamics donor)
SRC_FOR = {
    "W.npy": f"{_ZELDA}/5TTrlqAguhQ_chunk_0484/frames/run_ahead.png",
    "r.npy": f"{_ZELDA}/-BxyBxfDKA0_chunk_0292/frames/still.png",
    "l.npy": f"{_ZELDA}/5TTrlqAguhQ_chunk_0484/frames/run_ahead.png",
}

# group = (name, target, action, [lambdas]); steps/layers fixed unless noted
GROUPS = [
    ("snow_r", SNOW, "r.npy", [0.5, 0.8, 1.0]),   # snow + yaw-right (the showcase)
    ("snow_W", SNOW, "W.npy", [0.8]),             # snow + forward
    ("ski_r",  SKI,  "r.npy", [0.8, 1.0]),        # second snow scene
    ("robot_r", ROBOT, "r.npy", [0.8]),           # cross-check vs known target
]
STEPS, LO, HI = [0, 1], 0.25, 0.85


# ---- worker-side control --------------------------------------------------
def w_prepare(worker):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.prepare(worker.pipeline.get_module("transformer"))
    return INJECTOR.num_layers


def w_config(worker, lam, steps, lo, hi):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.lambda_max = lam
    INJECTOR.inject_steps = tuple(steps)
    INJECTOR.layer_lo, INJECTOR.layer_hi = lo, hi
    return (INJECTOR.lambda_max, INJECTOR.inject_steps)


def w_mode(worker, mode):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    {"record": INJECTOR.start_record, "replay": INJECTOR.start_replay,
     "off": INJECTOR.off}[mode]()
    return INJECTOR.mode


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def load_actions(action_path):
    raw = np.load(action_path, allow_pickle=True).item()
    kb = np.asarray(raw["keyboard"], dtype=np.float32)[:NUM_FRAMES, :4]
    mouse = np.asarray(raw["mouse"], dtype=np.float32)[:NUM_FRAMES, :]
    if kb.shape[0] < NUM_FRAMES:
        kb = np.concatenate([kb, np.zeros((NUM_FRAMES - kb.shape[0], 4), np.float32)])
        mouse = np.concatenate([mouse, np.zeros((NUM_FRAMES - mouse.shape[0], 2), np.float32)])
    return torch.from_numpy(kb), torch.from_numpy(mouse)


def rollout(gen, image_path, action_path, out_dir):
    kb, mouse = load_actions(action_path)
    grid = torch.tensor([(NUM_FRAMES + 3) // 4, H // 8, W // 8])
    t0 = time.time()
    gen.generate_video(
        prompt="", image_path=image_path,
        mouse_cond=mouse.unsqueeze(0), keyboard_cond=kb.unsqueeze(0),
        grid_sizes=grid, num_frames=NUM_FRAMES, height=H, width=W,
        num_inference_steps=NUM_STEPS, seed=SEED,
        output_path=str(out_dir), save_video=True)
    print(f"    [rollout] {out_dir.parent.name}/{out_dir.name}: "
          f"{time.time() - t0:.1f}s", flush=True)


def main():
    print(f"[init] loading {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False,
        dit_cpu_offload=False, vae_cpu_offload=False,
        text_encoder_cpu_offload=True, pin_cpu_memory=False)
    nlayers = rpc(gen, w_prepare)[0]
    print(f"[init] self-attn layers={nlayers}", flush=True)

    for name, tgt, action, lambdas in GROUPS:
        src = SRC_FOR[action]
        action_path = f"{_ACT}/{action}"
        gdir = OUT / f"sweep_{name}"
        print(f"\n##### {name}: tgt={Path(tgt).name} action={action} "
              f"lambdas={lambdas} #####", flush=True)

        rpc(gen, w_mode, "record")
        rollout(gen, src, action_path, gdir / "source")

        rpc(gen, w_mode, "off")
        rollout(gen, tgt, action_path, gdir / "baseline")

        for lam in lambdas:
            rpc(gen, w_config, lam, STEPS, LO, HI)
            rpc(gen, w_mode, "replay")
            tag = f"lam{int(round(lam * 100)):03d}"
            rollout(gen, tgt, action_path, gdir / tag)

    print("\n[done] sweep complete", flush=True)


if __name__ == "__main__":
    main()
