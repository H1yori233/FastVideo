"""exp52: feathered value-anchor vs hard-box. robot NF=297 (long validation).
hard06 (current best) vs feather{06,08,10}. Feather should let lambda go higher
without the background-pull haze. Judge by EYE on full-res frames; metric corroborates.
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
            m.sink_size = 1; m.local_attn_size = 6
    INJECTOR.prepare(tf)
    INJECTOR.grid_rows, INJECTOR.grid_cols = 30, 52
    INJECTOR.fg_box = BOX; INJECTOR.fg_soft = True; INJECTOR.fg_sigma = (5.0, 5.0)
    INJECTOR.start_fg_sink()
    INJECTOR.fg_boost = 4.0; INJECTOR.fg_qsupp = 6.0
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    return "init"


def w_set(worker, vhold, feather):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.fg_vhold = vhold
    INJECTOR.fg_vhold_feather = feather
    INJECTOR.fg_vhold_ramp = 0.0
    return (vhold, feather)


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
    for tag, vhold, feather in [("hard06", 0.6, False), ("feather06", 0.6, True),
                                ("feather08", 0.8, True), ("feather10", 1.0, True)]:
        print("[cfg]", tag, rpc(gen, w_set, vhold, feather)[0], flush=True)
        out = f"attn_injection_out/exp52_vfeather/{tag}"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=IMG, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {tag} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
