"""Corroborating eval: (1) CLIP as a 2nd identity backbone, (2) attractor-escape.

Attractor-escape: the clean baseline's FINAL-frame subject crop is the "Link pole"
(the prior the OOD subject drifts to). We measure DINO sim(frame_t, Link-pole) over
the rollout. clean should RISE toward 1 (falls into the attractor); ours should stay
LOW (escapes it). This directly measures the paper's core claim, complementary to
sim-to-anchor. Also reports CLIP sim-to-anchor to show the ID gain isn't DINO-specific.
"""
import argparse, json
import cv2, numpy as np, torch
import torch.nn.functional as F
import torchvision.transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, GR, GC = 480, 832, 30, 52
_N_DINO = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_N_CLIP = T.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])


def grid_to_px(b):
    r0, r1, c0, c1 = b
    return (round(r0/GR*H), round(r1/GR*H), round(c0/GC*W), round(c1/GC*W))


def crop_t(bgr, px, norm):
    r0, r1, c0, c1 = px
    rgb = cv2.cvtColor(bgr[r0:r1, c0:c1], cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.)
    t = F.interpolate(t.unsqueeze(0), (224, 224), mode="bilinear", align_corners=False)
    return norm(t.squeeze(0)).unsqueeze(0).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--crop", type=int, nargs=4, required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="attractor")
    args = ap.parse_args()
    px = grid_to_px(args.crop)

    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(DEV).eval()
    import open_clip
    clip_m, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    clip_m = clip_m.to(DEV).eval()

    def dfeat(bgr):
        with torch.no_grad(): return F.normalize(dino(crop_t(bgr, px, _N_DINO)), dim=-1)
    def cfeat(bgr):
        with torch.no_grad():
            return F.normalize(clip_m.encode_image(crop_t(bgr, px, _N_CLIP)).float(), dim=-1)

    a = cv2.resize(cv2.imread(args.anchor), (W, H))
    a_d, a_c = dfeat(a), cfeat(a)

    # Link pole = clean's final frame
    capc = cv2.VideoCapture(args.clean); nc = int(capc.get(7))
    capc.set(1, nc-1); _, link = capc.read(); capc.release()
    if link.shape[:2] != (H, W): link = cv2.resize(link, (W, H))
    link_d = dfeat(link)

    res = {}
    for tag, path in [("clean", args.clean), ("ours", args.ours)]:
        cap = cv2.VideoCapture(path); n = int(cap.get(7))
        xs, anc_d, anc_c, to_link = [], [], [], []
        for i in range(0, n, args.stride):
            cap.set(1, i); ok, fr = cap.read()
            if not ok: break
            if fr.shape[:2] != (H, W): fr = cv2.resize(fr, (W, H))
            d = dfeat(fr)
            xs.append(i)
            anc_d.append(float((d*a_d).sum())); anc_c.append(float((cfeat(fr)*a_c).sum()))
            to_link.append(float((d*link_d).sum()))
        cap.release()
        res[tag] = dict(x=xs, dino_anchor=anc_d, clip_anchor=anc_c, dino_link=to_link)
        print(f"{tag:6s} DINO->anchor {np.mean(anc_d):.3f}  CLIP->anchor {np.mean(anc_c):.3f}"
              f"  DINO->Link {np.mean(to_link):.3f} (final {np.mean(to_link[-5:]):.3f})",
              flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, c in [("clean", "tab:red"), ("ours", "tab:green")]:
        r = res[tag]
        ax[0].plot(r["x"], r["dino_anchor"], c=c, lw=2, label=f"{tag} DINO")
        ax[0].plot(r["x"], r["clip_anchor"], c=c, lw=1.3, ls="--", label=f"{tag} CLIP")
        ax[1].plot(r["x"], r["dino_link"], c=c, lw=2, label=tag)
    ax[0].set_title("Identity -> anchor (DINO solid, CLIP dashed)\nhigher=better")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("sim to anchor"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_title("Distance to Link attractor (sim to clean's end-state)\nhigher=fell into prior")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("sim to Link pole"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"attn_injection_out/_eval_{args.out}.png", dpi=115)
    json.dump(res, open(f"attn_injection_out/_eval_{args.out}.json", "w"))
    print("fig ->", f"attn_injection_out/_eval_{args.out}.png")


if __name__ == "__main__":
    main()
