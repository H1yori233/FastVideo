"""Recover walking motion while keeping identity: sink + larger cache window.

sink pins the first block (identity); a larger local_attn_size keeps a full
recent window alongside the sink so motion isn't starved. Sweep (sink, window).
"""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV = "/home/hal-kaiqin/FastVideo"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
TGT = f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp8_window")
H, W, NF = 480, 832, 117
# (sink_frames, window_frames)
CONFIGS = [(0, 6), (3, 6), (3, 9), (3, 12), (5, 12)]


def w_set(worker, sink, window):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    tf = worker.pipeline.get_module("transformer")
    n = 0
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = sink; m.local_attn_size = window; n += 1
    tf.local_attn_size = window
    stages_set = []
    for st in worker.pipeline.stages:
        if hasattr(st, "local_attn_size"):
            st.local_attn_size = window
            stages_set.append(type(st).__name__)
    return (sink, window, n, stages_set)


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def actions():
    raw = np.load(_ACT, allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:NF, :4])
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:NF, :])
    return kb, mo


def main():
    print(f"[init] {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    kb, mo = actions()
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])
    for sink, win in CONFIGS:
        print(f"[set] {rpc(gen, w_set, sink, win)[0]}", flush=True)
        t0 = time.time()
        gen.generate_video(prompt="", image_path=TGT, mouse_cond=mo.unsqueeze(0),
                           keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
                           height=H, width=W, num_inference_steps=4, seed=42,
                           output_path=str(OUT / f"s{sink}_w{win}"), save_video=True)
        print(f"    [rollout] s{sink}_w{win}: {time.time()-t0:.1f}s", flush=True)
    print("[done] exp8", flush=True)


if __name__ == "__main__":
    main()
