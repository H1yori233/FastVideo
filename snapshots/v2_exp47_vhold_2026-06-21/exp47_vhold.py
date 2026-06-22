"""exp47: fix long-rollout identity drift via SUBJECT-REGION value anchor.

New mechanism (fg_vhold): blend the current within-frame VALUES, inside the
subject box only, toward the frame-0 anchor values. Pins subject appearance/
identity over a long rollout and pulls the spawned back-glider region (absent in
frame 0) back toward the glider-free anchor — WITHOUT freezing routing/articulation
or touching the background. Distinct from the whole-frame value-hold dead end.

robot NF=297, op-A recipe. Sweep: ctrl(0) vs blend{0.3,0.6} vs adain(0.5).
"""
import time
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
        "WanGame2.1_1.3b_i2v/actions_801")
IMG = "/home/hal-kaiqin/FastVideo/assets/third-person/combine/robot_zelda_scene.jpg"
H, W, NF = 480, 832, 297
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
    INJECTOR.fg_qsupp = 6.0
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    return "init"


def w_set(worker, vhold, adain):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.fg_vhold = vhold
    INJECTOR.fg_vhold_adain = adain
    return (vhold, adain)


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def main():
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print("[init]", rpc(gen, w_init)[0], flush=True)
    raw = np.load(f"{_ACT}/W.npy", allow_pickle=True).item()
    kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:NF, :4])
    mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:NF, :])
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])

    configs = [
        ("blend08", 0.8, False),
        ("blend10", 1.0, False),
        ("adain08", 0.8, True),
    ]
    for tag, vhold, adain in configs:
        print("[cfg]", tag, rpc(gen, w_set, vhold, adain)[0], flush=True)
        out = f"attn_injection_out/exp47_vhold/{tag}"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=IMG, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {tag} {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
