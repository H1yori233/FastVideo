"""exp48: component ablation ladder (each rung adds ONE component).

  1 clean      sink0, injector OFF            (base model: frame-0 evicted)
  2 +sink      sink1, OFF                     (keep frame-0 in KV cache)
  3 +boost     sink1, fg_sink, boost4         (feathered subject anchor bias)
  4 +qsupp     ...    + qsupp6                (bg query-supp -> travel+articulate)
  5 +layergate ...    + layers 0.2-0.8        (= base recipe "ours")
  6 +vhold     ...    + fg_vhold 0.6          (= ours-full: subject value anchor)

robot, NF=201, action W. Eval with eval_identity.py (DINO id-sim + dynamism).
"""
import time
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator

MODEL = "/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_ACT = ("/home/hal-kaiqin/FastVideo/examples/training/finetune/"
        "WanGame2.1_1.3b_i2v/actions_801")
IMG = "/home/hal-kaiqin/FastVideo/assets/third-person/combine/robot_zelda_scene.jpg"
H, W, NF = 480, 832, 201
BOX = (14, 30, 21, 31)

# tag: (sink, on, boost, qsupp, lo, hi, vhold)
LADDER = [
    ("1_clean",     0, False, 0.0, 0.0, 0.0, 1.0, 0.0),
    ("2_sink",      1, False, 0.0, 0.0, 0.0, 1.0, 0.0),
    ("3_boost",     1, True,  4.0, 0.0, 0.0, 1.0, 0.0),
    ("4_qsupp",     1, True,  4.0, 6.0, 0.0, 1.0, 0.0),
    ("5_layergate", 1, True,  4.0, 6.0, 0.2, 0.8, 0.0),
    ("6_vhold",     1, True,  4.0, 6.0, 0.2, 0.8, 0.6),
]


def w_init(worker):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    INJECTOR.prepare(tf)
    INJECTOR.grid_rows, INJECTOR.grid_cols = 30, 52
    INJECTOR.fg_box = BOX
    INJECTOR.fg_soft = True
    INJECTOR.fg_sigma = (5.0, 5.0)
    return "init"


def w_cfg(worker, sink, on, boost, qsupp, lo, hi, vhold):
    from fastvideo.models.dits.matrixgame2.causal_model import (
        CausalMatrixGame2SelfAttention)
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf = worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m, CausalMatrixGame2SelfAttention):
            m.sink_size = sink
            m.local_attn_size = 6
    if not on:
        INJECTOR.off()
    else:
        INJECTOR.start_fg_sink()
        INJECTOR.fg_boost = boost
        INJECTOR.fg_qsupp = qsupp
        INJECTOR.layer_lo, INJECTOR.layer_hi = lo, hi
        INJECTOR.fg_vhold = vhold
    return f"sink{sink} on{on} b{boost} q{qsupp} L{lo}-{hi} v{vhold}"


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
    for tag, sink, on, boost, qsupp, lo, hi, vhold in LADDER:
        print("[cfg]", tag, rpc(gen, w_cfg, sink, on, boost, qsupp, lo, hi, vhold)[0],
              flush=True)
        out = f"attn_injection_out/exp48_ablation/{tag}"
        t0 = time.time()
        gen.generate_video(
            prompt="", image_path=IMG, mouse_cond=mo.unsqueeze(0),
            keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
            height=H, width=W, num_inference_steps=4, seed=42,
            output_path=out, save_video=True)
        print(f"[done] {tag} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
