"""Final: probe sink2 (motion/identity tradeoff) + render clean long sink3 walk."""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV = "/home/hal-kaiqin/FastVideo"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
TGT = f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp12_final")
H, W = 480, 832
# (sink, window, num_frames, tag)
JOBS = [(2, 6, 117, "sink2_117"), (3, 6, 201, "sink3_201"), (2, 9, 201, "sink2_w9_201")]


def w_set(worker, sink, window):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    tf = worker.pipeline.get_module("transformer")
    n = 0
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = sink; m.local_attn_size = window; n += 1
    tf.local_attn_size = window
    for st in worker.pipeline.stages:
        if hasattr(st, "local_attn_size"):
            st.local_attn_size = window
    return (sink, window, n)


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def actions(nf):
    raw = np.load(_ACT, allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:nf, :4])
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:nf, :])
    return kb, mo


def main():
    print(f"[init] {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    for sink, win, nf, tag in JOBS:
        print(f"[set] {rpc(gen, w_set, sink, win)[0]}", flush=True)
        kb, mo = actions(nf)
        grid = torch.tensor([(nf + 3) // 4, H // 8, W // 8])
        t0 = time.time()
        gen.generate_video(prompt="", image_path=TGT, mouse_cond=mo.unsqueeze(0),
                           keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=nf,
                           height=H, width=W, num_inference_steps=4, seed=42,
                           output_path=str(OUT / tag), save_video=True)
        print(f"    [rollout] {tag}: {time.time()-t0:.1f}s", flush=True)
    print("[done] exp12", flush=True)


if __name__ == "__main__":
    main()
