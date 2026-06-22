"""exp54: validate the value-anchor on MULTIPLE long videos (297f).
Generate +value-anchor (feathered vhold0.6) for wukong & mc to pair against the
base-recipe 297f runs already in exp46_longshow (robot has feather06 in exp52).
Then before/after is judged by EYE across 3 subjects.
"""
import time
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
        "WanGame2.1_1.3b_i2v/actions_801")
A = "/home/hal-kaiqin/FastVideo/assets/third-person"
H, W, NF = 480, 832, 297
SUBJ = {
    "wukong":  (f"{A}/combine/wukong_zelda_scene.jpg", (15, 29, 16, 29)),
    "mc":      (f"{A}/mc_third_person.jpg",            (12, 27, 23, 33)),
}


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
    INJECTOR.fg_soft = True; INJECTOR.fg_sigma = (5.0, 5.0)
    INJECTOR.start_fg_sink()
    INJECTOR.fg_boost = 4.0; INJECTOR.fg_qsupp = 6.0
    INJECTOR.layer_lo, INJECTOR.layer_hi = 0.2, 0.8
    INJECTOR.fg_vhold = 0.6; INJECTOR.fg_vhold_feather = True
    return "init"


def w_box(worker, box):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.fg_box = tuple(box)
    return str(box)


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
    for subj, (img, box) in SUBJ.items():
        print("[cfg]", subj, rpc(gen, w_box, box)[0], flush=True)
        out = f"attn_injection_out/exp54_valid/{subj}_vhold"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=img, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {subj} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
