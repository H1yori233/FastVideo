"""Synthesis: sink (robot identity) + cross-frame routing injection (Link motion).

sink_size=3 pins the robot anchor (first block) -> identity. Cross-frame routing
injection borrows Link's recent-frame motion correspondences (the injector's
cross columns exclude the sink, so the anchor is never overwritten). V stays the
target's -> appearance robot. Tests whether this makes the robot WALK while
staying a robot.
"""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV = "/home/hal-kaiqin/FastVideo"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
SRC = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/5TTrlqAguhQ_chunk_0484/frames/run_ahead.png"
TGT = f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp10_pose")
H, W, NF = 480, 832, 117
SINK, WIN = 3, 9
LAMBDAS = [0.3, 0.6, 1.0]
STEPS, LO, HI = [0, 1], 0.0, 1.0  # all layers (forward motion is global)


def w_setup(worker, sink, window):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    n = 0
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = sink; m.local_attn_size = window; n += 1
    tf.local_attn_size = window
    for st in worker.pipeline.stages:
        if hasattr(st, "local_attn_size"):
            st.local_attn_size = window
    INJECTOR.prepare(tf)
    return (sink, window, n, INJECTOR.num_layers)


def w_config(worker, lam, lamw, steps, lo, hi):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.lambda_max = lam; INJECTOR.lambda_within = lamw; INJECTOR.inject_steps = tuple(steps)
    INJECTOR.layer_lo, INJECTOR.layer_hi = lo, hi
    return lam


def w_mode(worker, mode):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    {"record": INJECTOR.start_record, "replay": INJECTOR.start_replay, "off": INJECTOR.off}[mode]()
    return INJECTOR.mode


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def actions():
    raw = np.load(_ACT, allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:NF, :4])
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:NF, :])
    return kb, mo


def rollout(gen, img, out):
    kb, mo = actions()
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])
    t0 = time.time()
    gen.generate_video(prompt="", image_path=img, mouse_cond=mo.unsqueeze(0),
                       keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
                       height=H, width=W, num_inference_steps=4, seed=42,
                       output_path=str(out), save_video=True)
    print(f"    [rollout] {out.name}: {time.time()-t0:.1f}s", flush=True)


def main():
    print(f"[init] {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print(f"[setup] {rpc(gen, w_setup, SINK, WIN)[0]}", flush=True)

    # baseline: sink only, no injection
    rpc(gen, w_mode, "off")
    rollout(gen, TGT, OUT / "baseline_sink3")

    # record Link source motion (under same sink/window)
    rpc(gen, w_mode, "record")
    rollout(gen, SRC, OUT / "source")

    for lam in LAMBDAS:
        rpc(gen, w_config, 0.0, lam, STEPS, LO, HI)
        rpc(gen, w_mode, "replay")
        rollout(gen, TGT, OUT / f"inj_lam{int(lam*100):03d}")
    print("[done] exp9", flush=True)


if __name__ == "__main__":
    main()
