"""Build a contact sheet for one sweep group: rows = variants, cols = frames.

Usage: python make_contact_sheet.py <group_dir> [frame indices...]
Saves <group_dir>/contact.png (downscaled) for visual inspection.
"""
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import imageio

ROW_ORDER = ["source", "baseline", "lam050", "lam080", "lam100"]


def label(img, text):
    # crude top-left dark bar with no font dep: just darken a strip
    img = img.copy()
    img[:14, :len(text) * 7 + 4] = (img[:14, :len(text) * 7 + 4] * 0.2).astype(img.dtype)
    return img


def main():
    gdir = Path(sys.argv[1])
    frames = [int(x) for x in sys.argv[2:]] or [0, 14, 28, 42, 56]
    rows = []
    for name in ROW_ORDER:
        mp4 = gdir / name / "output.mp4"
        if not mp4.exists():
            continue
        vid = iio.imread(mp4)
        sel = [vid[min(f, len(vid) - 1)] for f in frames]
        strip = np.concatenate(sel, axis=1)
        # downscale x2 for compactness
        strip = strip[::2, ::2]
        strip = label(strip, name)
        rows.append(strip)
    sheet = np.concatenate(rows, axis=0).astype(np.uint8)
    out = gdir / "contact.png"
    imageio.imwrite(out, sheet)
    print(f"wrote {out}  shape={sheet.shape}  rows={[r for r in ROW_ORDER if (gdir/r/'output.mp4').exists()]}  frames={frames}")


if __name__ == "__main__":
    main()
