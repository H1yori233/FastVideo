"""Explicit motion transfer: warp the robot anchor along Link's optical flow.

Attention injection cannot synthesize off-manifold gait (proved empirically).
Model-independent alternative: take Link source's per-frame optical flow and
propagate the robot anchor through it. Guarantees the robot identity (it IS the
warped anchor) and Link's exact motion; cost is non-generative warping (no newly
revealed content; border stretching).
"""
import numpy as np, cv2, imageio.v3 as iio, imageio
from pathlib import Path

SRC = "attn_injection_out/exp10_pose/source/output.mp4"          # Link walking
ANCHOR = "/home/hal-kaiqin/FastVideo/assets/third-person/combine/robot_zelda_scene.jpg"
SINK3 = "attn_injection_out/exp10_pose/baseline_sink3/output.mp4"  # generative robot
OUT = Path("attn_injection_out/flow_warp")
OUT.mkdir(parents=True, exist_ok=True)


def avgmag(frames):
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    return float(np.mean([np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 21, 3, 5, 1.2, 0), axis=2).mean()
                          for a, b in zip(g[:-1], g[1:])]))


def main():
    link = iio.imread(SRC)                       # [T,H,W,3]
    T, H, W, _ = link.shape
    anchor = cv2.resize(iio.imread(ANCHOR)[..., :3], (W, H))
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    out = [anchor.copy()]
    prev = anchor.copy()
    lg = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in link]
    for t in range(1, T):
        fl = cv2.calcOpticalFlowFarneback(lg[t - 1], lg[t], None, .5, 3, 21, 3, 5, 1.2, 0)
        # backward map: sample prev at (x - flow) so content moves by +flow like Link
        mapx = (gx - fl[..., 0]).astype(np.float32)
        mapy = (gy - fl[..., 1]).astype(np.float32)
        prev = cv2.remap(prev, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append(prev.copy())
    out = np.stack(out).astype(np.uint8)
    imageio.mimwrite(OUT / "robot_flowwarp.mp4", out, fps=14, quality=7)
    print(f"warp motion={avgmag(out):.2f}  link={avgmag(link):.2f}", flush=True)

    # side-by-side: link | sink3(generative) | flow-warp(robot)
    sink3 = iio.imread(SINK3)
    n = min(len(link), len(sink3), len(out))
    def lab(v, t):
        v = v[:n].copy()
        for f in v: f[:18, :len(t) * 9 + 6] = (f[:18, :len(t) * 9 + 6] * 0.15).astype(f.dtype)
        return v
    strip = np.concatenate([lab(link, "link motion"), lab(sink3, "sink3 robot gen"),
                            lab(out, "flow-warp robot")], axis=2).astype(np.uint8)
    imageio.mimwrite(OUT / "sbs_flowwarp.mp4", strip, fps=14, quality=7)
    print("wrote sbs_flowwarp.mp4", flush=True)


if __name__ == "__main__":
    main()
