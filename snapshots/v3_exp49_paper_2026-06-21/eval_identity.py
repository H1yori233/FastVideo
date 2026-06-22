"""Quantitative identity-preservation eval for OOD world-model rollouts.

Metric: DINO ViT-B/16 self-supervised feature cosine similarity between the
SUBJECT-REGION crop at frame t and the ANCHOR image's subject crop. DINO features
capture appearance/identity robustly. A good (identity-preserving) rollout keeps
this similarity HIGH and FLAT; a drifting one (-> Link prior) DECAYS.

Reports per video: mean sim, final-frame sim, and drift slope (linear fit, per
100 frames; negative = losing identity). Saves a curve figure + JSON.

Usage:
  python eval_identity.py --anchor <anchor.jpg> --crop r0 r1 c0 c1 \
    --videos ours:path1.mp4 clean:path2.mp4 --out tag
(crop in 30x52 token-grid coords; converted to pixels for 480x832.)
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
import torchvision.transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, GR, GC = 480, 832, 30, 52
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_dino():
    m = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(DEV).eval()
    return m


def grid_to_px(box):
    r0, r1, c0, c1 = box
    return (round(r0 / GR * H), round(r1 / GR * H),
            round(c0 / GC * W), round(c1 / GC * W))


def feat(model, bgr, px):
    r0, r1, c0, c1 = px
    crop = bgr[r0:r1, c0:c1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.)
    t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear",
                      align_corners=False)
    t = _NORM(t.squeeze(0)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        f = model(t)  # [1, 768] CLS
    return F.normalize(f, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--crop", type=int, nargs=4, required=True,
                    help="r0 r1 c0 c1 of subject in 30x52 grid")
    ap.add_argument("--videos", nargs="+", required=True, help="tag:path ...")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="ident")
    args = ap.parse_args()
    px = grid_to_px(args.crop)
    model = load_dino()

    a_bgr = cv2.imread(args.anchor)
    a_bgr = cv2.resize(a_bgr, (W, H))
    a_feat = feat(model, a_bgr, px)

    results = {}
    for spec in args.videos:
        tag, path = spec.split(":", 1)
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        xs, sims, feats = [], [], []
        for i in range(0, n, args.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[:2] != (H, W):
                fr = cv2.resize(fr, (W, H))
            fe = feat(model, fr, px)
            s = float((fe * a_feat).sum().item())
            xs.append(i); sims.append(s); feats.append(fe.squeeze(0).cpu().numpy())
        cap.release()
        xs, sims = np.array(xs), np.array(sims)
        slope = float(np.polyfit(xs, sims, 1)[0] * 100) if len(xs) > 1 else 0.0
        # dynamism: mean consecutive-sampled-frame feature L2 (subject still moves?
        # a frozen rollout -> ~0). Proves identity isn't held by freezing.
        ff = np.array(feats)  # [N, D] unit-norm
        dyn = float(np.linalg.norm(ff[1:] - ff[:-1], axis=1).mean()) if len(ff) > 1 else 0.0
        results[tag] = dict(x=xs.tolist(), sim=sims.tolist(),
                            mean=float(sims.mean()), final=float(sims[-5:].mean()),
                            slope_per100=slope, dynamism=dyn)
        print(f"{tag:14s} mean={sims.mean():.3f} final={sims[-5:].mean():.3f} "
              f"slope/100f={slope:+.4f} dyn={dyn:.3f}", flush=True)

    # curve figure (matplotlib-free: draw on a canvas)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.2))
        for tag, r in results.items():
            plt.plot(r["x"], r["sim"], label=f"{tag} (μ={r['mean']:.2f})", lw=2)
        plt.xlabel("frame"); plt.ylabel("DINO identity sim to anchor")
        plt.title("Identity preservation over rollout"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout()
        out_png = f"attn_injection_out/_eval_{args.out}.png"
        plt.savefig(out_png, dpi=110); print("fig ->", out_png)
    except Exception as e:
        print("plot skipped:", e)
    Path(f"attn_injection_out/_eval_{args.out}.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
