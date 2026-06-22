"""exp46: LONG-rollout proof across diverse OOD examples (NF=297).

Proves the training-free Foreground-Anchored Attention recipe on a LONG rollout
and MANY examples:
  - 4 humanoids w/ recipe (robot, wukong, genshin, minecraft) -> identity held +
    articulate + travel where a plain rollout drifts to Link.
  - robot CLEAN baseline (sink0, injector off) -> drifts to Link (before/after).
  - 2 non-humanoids clean (car, robot-dog) -> identity held already (no fg needed).
One model load; per-case INJECTOR config; action W (forward) so travel is visible.
"""
import time
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
        "WanGame2.1_1.3b_i2v/actions_801")
A = "/home/hal-kaiqin/FastVideo/assets/third-person"
H, W, NF = 480, 832, 297

# (tag, image, box | None=clean, action)
CASES = [
    ("robot_fg",   f"{A}/combine/robot_zelda_scene.jpg",     (14, 30, 21, 31), "W"),
    ("robot_clean", f"{A}/combine/robot_zelda_scene.jpg",    None,             "W"),
    ("wukong_fg",  f"{A}/combine/wukong_zelda_scene.jpg",    (15, 29, 16, 29), "W"),
    ("genshin_fg", f"{A}/genshin.png",                       (12, 27, 23, 32), "W"),
    ("mc_fg",      f"{A}/mc_third_person.jpg",               (12, 27, 23, 33), "W"),
    ("car_clean",  f"{A}/combine/car_zelda_scene.jpg",       None,             "W"),
    ("dog_clean",  f"{A}/combine/robot_dog_zelda_scene.jpg", None,             "W"),
]


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
    INJECTOR.fg_soft = True
    INJECTOR.fg_sigma = (5.0, 5.0)
    INJECTOR.fg_boost = 4.0
    INJECTOR.fg_qsupp = 6.0
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    return "init"


def w_set(worker, box):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    if box is None:
        INJECTOR.off()
        return "clean(off)"
    INJECTOR.fg_box = tuple(box)
    INJECTOR.start_fg_sink()
    return f"fg{tuple(box)}"


def rpc(gen, fn, *a):
    return gen.executor.collective_rpc(cloudpickle.dumps(fn), args=a)


def main():
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    print("[init]", rpc(gen, w_init)[0], flush=True)
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])

    for tag, img, box, act in CASES:
        print("[cfg]", tag, rpc(gen, w_set, box)[0], flush=True)
        raw = np.load(f"{_ACT}/{act}.npy", allow_pickle=True).item()
        kb = torch.from_numpy(np.asarray(raw["keyboard"], np.float32)[:NF, :4])
        mo = torch.from_numpy(np.asarray(raw["mouse"], np.float32)[:NF, :])
        out = f"attn_injection_out/exp46_longshow/{tag}"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=img, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {tag} {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
