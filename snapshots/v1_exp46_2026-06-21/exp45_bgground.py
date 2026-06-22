"""exp45: late background re-grounding via frame-scheduled qsupp decay.

Hypothesis: the late-rollout background Zelda-hallucination is caused by the
soft query-suppression fully detaching background queries from the frame-0
anchor. If we DECAY qsupp late, bg queries re-attend the anchor's background
appearance -> re-grounds style (less Link-gear/creature spawn) without a hard
geometric tether. Run at operating point B (low base qsupp + kb x2) so the
subject keeps articulating from the action drive (not from qsupp).

Configs (robot, NF=201): ctrl decay=0  vs  decay {0.01, 0.02}, floor 0.5.
"""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT_DIR = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
            "WanGame2.1_1.3b_i2v/actions_801")
IMAGE = "/home/hal-kaiqin/FastVideo/assets/third-person/combine/robot_zelda_scene.jpg"
H, W = 480, 832
NF = 201
KB_SCALE = 1.0
BOX = (14, 30, 21, 31)


def w_init(worker):
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
    INJECTOR.fg_box = BOX
    INJECTOR.fg_soft = True
    INJECTOR.fg_sigma = (5.0, 5.0)
    INJECTOR.start_fg_sink()
    INJECTOR.fg_boost = 4.0
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    return "init"


def w_set(worker, qsupp, decay, floor):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.fg_qsupp = qsupp
    INJECTOR.fg_qsupp_decay = decay
    INJECTOR.fg_qsupp_floor = floor
    return (qsupp, decay, floor)


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def main():
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print("[init]", rpc(gen, w_init)[0], flush=True)

    raw = np.load(f"{_ACT_DIR}/W.npy", allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:NF, :4]) * KB_SCALE
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:NF, :])
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])

    # operating point A (high qsupp, traveling world) — where the severe Zelda-
    # gear hallucination lives. Test whether late qsupp-decay re-grounds the bg.
    configs = [
        ("A_ctrl_d0", 6.0, 0.0, 0.0),
        ("A_dec01", 6.0, 0.01, 2.5),
        ("A_dec02", 6.0, 0.02, 2.5),
    ]
    for tag, qsupp, decay, floor in configs:
        print("[cfg]", tag, rpc(gen, w_set, qsupp, decay, floor)[0], flush=True)
        out = f"attn_injection_out/exp45_bgground/{tag}"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=IMAGE, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {tag} {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
