"""Probe which (model, anchor) combos animate vs freeze. Baseline only."""
import sys, time
from pathlib import Path
import numpy as np, cv2, torch, imageio.v3 as iio
from fastvideo import VideoGenerator

MODEL = sys.argv[1]
_FV = "/home/hal-kaiqin/FastVideo"
_ACT = f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
ANCHORS = {
    "src_runahead": f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/5TTrlqAguhQ_chunk_0484/frames/run_ahead.png",
    "natzelda": f"{_FV}/assets/third-person/combine/zelda_natural_scene.jpg",
    "robotzelda": f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg",
    "wukong": f"{_FV}/assets/third-person/wukong_832x480.jpg",
    "snow": f"{_FV}/assets/images/mixkit-curve-on-a-snowy-forest-road-3317.png",
}
H, W, NF = 480, 832, 117
OUT = Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/freeze_probe")


def actions():
    raw = np.load(_ACT, allow_pickle=True).item()
    kb = np.asarray(raw["keyboard"], np.float32)[:NF, :4]
    mo = np.asarray(raw["mouse"], np.float32)[:NF, :]
    return torch.from_numpy(kb), torch.from_numpy(mo)


def avgflow(mp4):
    v = iio.imread(mp4); g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in v]
    m = [np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 21, 3, 5, 1.2, 0), axis=2).mean()
         for a, b in zip(g[:-1], g[1:])]
    return float(np.mean(m))


def main():
    tag = MODEL.rstrip("/").split("/")[-1]
    print(f"[init] {MODEL}", flush=True)
    gen = VideoGenerator.from_pretrained(
        MODEL, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    kb, mo = actions()
    grid = torch.tensor([(NF + 3) // 4, H // 8, W // 8])
    for name, img in ANCHORS.items():
        od = OUT / tag / name
        t0 = time.time()
        gen.generate_video(prompt="", image_path=img, mouse_cond=mo.unsqueeze(0),
                           keyboard_cond=kb.unsqueeze(0), grid_sizes=grid, num_frames=NF,
                           height=H, width=W, num_inference_steps=4, seed=42,
                           output_path=str(od), save_video=True)
        f = avgflow(od / "output.mp4")
        print(f"[flow] {tag:28s} {name:14s} avg_mag={f:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
