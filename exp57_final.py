"""exp57: canonical CURRENT-BEST set for review. The validated recipe exactly:
sink1/win6, feathered boost4, qsupp6, mid-layer-gate 0.2-0.8, sigma(5,5),
HARD-box value-anchor vhold0.6 (feather/vmatch OFF). Humanoids use the recipe;
non-humanoids (car/dog) run clean (no Link attractor). NF=297, action W, seed42.
"""
import time
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
        "WanGame2.1_1.3b_i2v/actions_801")
A = "/home/hal-kaiqin/FastVideo/assets/third-person"
H, W, NF = 480, 832, 297

# tag: (image, box | None=clean non-humanoid)
CASES = [
    ("robot",   f"{A}/combine/robot_zelda_scene.jpg",     (14, 30, 21, 31)),
    ("wukong",  f"{A}/combine/wukong_zelda_scene.jpg",    (15, 29, 16, 29)),
    ("genshin", f"{A}/genshin.png",                       (12, 27, 23, 32)),
    ("mc",      f"{A}/mc_third_person.jpg",               (12, 27, 23, 33)),
    ("car",     f"{A}/combine/car_zelda_scene.jpg",       None),
    ("dog",     f"{A}/combine/robot_dog_zelda_scene.jpg", None),
]


def w_init(worker):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.prepare(worker.pipeline.get_module("transformer"))
    INJECTOR.grid_rows, INJECTOR.grid_cols = 30, 52
    INJECTOR.fg_soft = True; INJECTOR.fg_sigma = (5.0, 5.0)
    return "init"


def w_cfg(worker, box):
    from fastvideo.models.dits.matrixgame2.causal_model import (
        CausalMatrixGame2SelfAttention)
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    on = box is not None
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = 1 if on else 0; m.local_attn_size = 6
    if not on:
        INJECTOR.off(); return "clean"
    INJECTOR.fg_box = tuple(box)
    INJECTOR.start_fg_sink()
    INJECTOR.fg_boost = 4.0; INJECTOR.fg_qsupp = 6.0
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    INJECTOR.fg_vhold = 0.6
    INJECTOR.fg_vhold_feather = False; INJECTOR.fg_vmatch = 0.0
    INJECTOR.fg_vhold_ramp = 0.0
    return f"recipe{tuple(box)}"


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
    for tag, img, box in CASES:
        print("[cfg]", tag, rpc(gen, w_cfg, box)[0], flush=True)
        out = f"attn_injection_out/exp57_final/{tag}"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=img, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {tag} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
