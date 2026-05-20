"""Stage 1 smoke test for SVIRawVideoDataset.

Run from repo root:
    python -m fastvideo.exploration.smoke_dataset
or directly:
    python .agents/exploration/svi-training/smoke_dataset.py

Asserts shape/dtype/range of each output field on the SVI toy_train dataset
and compares against upstream TextVideoDataset_onestage on identical seed
indices (when the upstream repo is available on disk).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fastvideo.train.datasets.svi_raw_video import SVIRawVideoDataset  # noqa: E402

TOY_PATH = REPO / "Stable-Video-Infinity" / "data" / "toy_train" / "svi-film-shot"
H, W, T = 480, 832, 81


def check_shapes(sample: dict) -> None:
    assert isinstance(sample["text"], str) and sample["text"], f"bad text: {sample['text']!r}"
    video = sample["video"]
    assert isinstance(video, torch.Tensor), type(video)
    assert video.shape == (3, T, H, W), video.shape
    assert video.dtype == torch.float32, video.dtype
    assert -1.05 <= video.min().item() <= -0.5 and 0.5 <= video.max().item() <= 1.05, (video.min(), video.max())

    refs = sample["first_ref_frames"]
    assert isinstance(refs, list) and len(refs) == 12, len(refs)
    for r in refs:
        assert isinstance(r, torch.Tensor), type(r)
        assert r.shape == (H, W, 3), r.shape
        assert r.dtype == torch.uint8, r.dtype

    rref = sample["random_ref_frame"]
    assert rref.shape == (H, W, 3), rref.shape
    assert rref.dtype == torch.uint8, rref.dtype

    assert isinstance(sample["path"], str) and sample["path"].endswith((".mp4", ".avi", ".mov", ".mkv"))


def parity_against_upstream(ds: SVIRawVideoDataset, idx: int) -> None:
    """Compare against upstream TextVideoDataset_onestage on the same index/seed.

    Skips if upstream is not importable (e.g. running outside the SVI env).
    """
    upstream_root = REPO / "Stable-Video-Infinity"
    if not upstream_root.exists():
        print("[parity] upstream not on disk, skipping")
        return
    sys.path.insert(0, str(upstream_root))
    try:
        # train_svi.py also defines the dataset; we import the class only.
        # Importing the module pulls in heavy deps (Lightning, diffsynth), so we
        # quote-source the dataset class instead. Practical fast path:
        from train_svi import TextVideoDataset_onestage  # type: ignore
    except Exception as exc:
        print(f"[parity] upstream import failed: {exc}; skipping")
        return

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    ours = ds[idx]

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    up_ds = TextVideoDataset_onestage(
        base_path=str(TOY_PATH),
        metadata_path="",
        max_num_frames=T,
        frame_interval=1,
        num_frames=T,
        height=H,
        width=W,
    )
    up_sample = up_ds[idx]

    print(f"[parity] video diff: {(ours['video'] - up_sample['video']).abs().max().item():.4e}")
    print(f"[parity] text match: {ours['text'] == up_sample['text']}")


def main() -> None:
    if not TOY_PATH.exists():
        raise FileNotFoundError(f"toy dataset missing at {TOY_PATH}")

    ds = SVIRawVideoDataset(
        base_path=str(TOY_PATH),
        num_frames=T,
        height=H,
        width=W,
        frame_interval=1,
        shuffle=True,
        seed=0,
    )
    print(f"dataset size: {len(ds)}")
    print(f"first item path: {ds.video_list[0].path}")

    random.seed(7)
    for i in range(5):
        s = ds[i]
        check_shapes(s)
        print(f"[ok] idx={i} text={s['text'][:50]!r} video={tuple(s['video'].shape)} "
              f"ref0={tuple(s['first_ref_frames'][0].shape)} rref={tuple(s['random_ref_frame'].shape)}")

    parity_against_upstream(ds, idx=0)
    print("OK")


if __name__ == "__main__":
    main()
