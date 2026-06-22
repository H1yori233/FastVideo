"""Value-hold: free rollout (sink0, vigorous motion) + blend within-frame V
toward the robot anchor -> hold appearance while leaving structure/motion free.
Tests whether appearance and motion decouple (the goal)."""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV = "/home/hal-kaiqin/FastVideo"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
TGT = f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp13_vhold")
H, W, NF = 480, 832, 117
LAMV = [0.5, 0.8, 1.0]


def w_setup(worker):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    n = 0
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = 0; m.local_attn_size = 6; n += 1  # free (no sink) -> motion
    tf.local_attn_size = 6
    for st in worker.pipeline.stages:
        if hasattr(st, "local_attn_size"):
            st.local_attn_size = 6
    INJECTOR.prepare(tf)
    INJECTOR.inject_steps = (0, 1, 2, 3)   # hold appearance at every denoise step
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.0, 1.0  # all layers
    return n


def w_lamv(worker, lamv):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.lambda_v = lamv
    return lamv


def w_mode(worker, mode):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    if mode == "anchor_hold":
        INJECTOR.start_anchor_hold()
    else:
        INJECTOR.off()
    return INJECTOR.mode


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def actions():
    raw = np.load(_ACT, allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:NF, :4])
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:NF, :])
    return kb, mo


def rollout(gen, out):
    kb, mo = actions()
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])
    t0 = time.time()
    gen.generate_video(prompt="", image_path=TGT, mouse_cond=mo.unsqueeze(0),
                       keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
                       height=H, width=W, num_inference_steps=4, seed=42,
                       output_path=str(out), save_video=True)
    print(f"    [rollout] {out.name}: {time.time()-t0:.1f}s", flush=True)


def main():
    print(f"[init] {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print(f"[setup] layers={rpc(gen, w_setup)[0]}", flush=True)

    rpc(gen, w_mode, "off")
    rollout(gen, OUT / "baseline_sink0")  # free, drifts to Link

    for lamv in LAMV:
        rpc(gen, w_lamv, lamv)
        rpc(gen, w_mode, "anchor_hold")
        rollout(gen, OUT / f"vhold_{int(lamv*100):03d}")
    print("[done] exp13", flush=True)


if __name__ == "__main__":
    main()
