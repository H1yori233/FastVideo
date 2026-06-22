"""Injection on the longlive model (which animates OOD), 117 frames.

Per action: record Zelda source once; per target: baseline + injected at each
lambda. longlive animates OOD targets, so cross-frame routing injection now has
motion content to reshape -> we can test whether it aligns the target's camera
response to the calibrated source.
"""
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch

from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV = "/home/hal-kaiqin/FastVideo"
_ZELDA = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801"
_COMB = f"{_FV}/assets/third-person/combine"
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out")

H, W = 480, 832
NUM_FRAMES = 117
NUM_STEPS = 4
SEED = 42

SRC_FOR = {
    "W.npy": f"{_ZELDA}/5TTrlqAguhQ_chunk_0484/frames/run_ahead.png",
    "r.npy": f"{_ZELDA}/-BxyBxfDKA0_chunk_0292/frames/still.png",
}
TARGETS = {
    "robotzelda": f"{_COMB}/robot_zelda_scene.jpg",   # snowy Zelda composite (animates 0.69)
    "natzelda": f"{_COMB}/zelda_natural_scene.jpg",
    "wukong": f"{_FV}/assets/third-person/wukong_832x480.jpg",
}
ACTIONS = ["r.npy", "W.npy"]
LAMBDAS = [0.5, 1.0]
STEPS, LO, HI = [0, 1], 0.25, 0.85


def w_prepare(worker):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.prepare(worker.pipeline.get_module("transformer"))
    return INJECTOR.num_layers


def w_config(worker, lam, steps, lo, hi):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.lambda_max = lam
    INJECTOR.inject_steps = tuple(steps)
    INJECTOR.layer_lo, INJECTOR.layer_hi = lo, hi
    return INJECTOR.lambda_max


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
    print(f"    [rollout] {out_dir.name}: {time.time() - t0:.1f}s", flush=True)


def main():
    print(f"[init] loading {MODEL}  NUM_FRAMES={NUM_FRAMES}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False,
        dit_cpu_offload=False, vae_cpu_offload=False,
        text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print(f"[init] self-attn layers={rpc(gen, w_prepare)[0]}", flush=True)

    for action in ACTIONS:
        a = action.replace(".npy", "")
        gdir = OUT / f"exp4_{a}"
        ap = f"{_ACT}/{action}"
        print(f"\n##### action={action} #####", flush=True)
        rpc(gen, w_mode, "record")
        rollout(gen, SRC_FOR[action], ap, gdir / "source")

        for tname, tgt in TARGETS.items():
            rpc(gen, w_mode, "off")
            rollout(gen, tgt, ap, gdir / f"{tname}_baseline")
            for lam in LAMBDAS:
                rpc(gen, w_config, lam, STEPS, LO, HI)
                rpc(gen, w_mode, "replay")
                rollout(gen, tgt, ap, gdir / f"{tname}_lam{int(lam*100):03d}")

    print("\n[done] exp4 complete", flush=True)


if __name__ == "__main__":
    main()
