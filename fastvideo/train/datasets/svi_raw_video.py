# SPDX-License-Identifier: Apache-2.0
"""Raw-video dataset for Stable-Video-Infinity training.

Mirrors the upstream ``TextVideoDataset_onestage`` from
``Stable-Video-Infinity/train_svi.py`` so per-step inputs to the LoRA
fine-tune match what the official training script feeds the model. The new
FastVideo training stack normally consumes pre-extracted parquet latents,
but SVI needs raw pixels for the CLIP image encoder, so this dataset
returns raw frames and the method does VAE/CLIP encoding on the fly.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, v2

from fastvideo.logger import init_logger

logger = init_logger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class SVIVideoItem:
    path: str
    description: str
    category: str


class SVIRawVideoDataset(Dataset):
    """Returns raw frames + 12 reference frames + 1 random reference frame.

    Per-sample dict:
        text:              caption string
        video:             float tensor [3, T, H, W] normalized to [-1, 1]
        first_ref_frames:  list of 12 uint8 tensors [H, W, 3] (post-resize)
        random_ref_frame:  uint8 tensor [H, W, 3] (post-resize)
        path:              source video path

    Layout of ``base_path`` (matches upstream):
        - ``base_path/<category>/<category>.csv`` with columns
          ``Filename, Video Description`` plus video files in the same dir
        - or, a flat directory of video files (caption defaults to a stub)
        - or, a single video file path
    """

    def __init__(
        self,
        base_path: str,
        *,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        frame_interval: int = 1,
        max_num_frames: int | None = None,
        num_ref_frames: int = 12,
        crop_jitter_frac: float = 1.0 / 14.0,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sample_fps = frame_interval
        self.max_frames = max_num_frames if max_num_frames is not None else num_frames
        self.num_ref_frames = num_ref_frames
        self.crop_jitter_frac = crop_jitter_frac

        self.video_list: list[SVIVideoItem] = []
        self._scan(base_path)
        if not self.video_list:
            raise FileNotFoundError(f"SVIRawVideoDataset: no videos found under {base_path!r}")

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.video_list)

        self.frame_process = v2.Compose([
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        logger.info("SVIRawVideoDataset: %d videos under %s", len(self.video_list), base_path)

    def _scan(self, base_path: str) -> None:
        if os.path.isfile(base_path):
            if base_path.lower().endswith(VIDEO_EXTENSIONS):
                self.video_list.append(SVIVideoItem(path=base_path, description="The video", category="single"))
            return

        if not os.path.isdir(base_path):
            return

        subdirs = sorted(d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)))
        if subdirs:
            for subdir in subdirs:
                subdir_path = os.path.join(base_path, subdir)
                csv_file = os.path.join(subdir_path, f"{subdir}.csv")
                descriptions = self._read_csv(csv_file) if os.path.exists(csv_file) else {}
                default_desc = f"A video from {subdir} category"
                for file in sorted(os.listdir(subdir_path)):
                    if file.lower().endswith(VIDEO_EXTENSIONS):
                        self.video_list.append(
                            SVIVideoItem(
                                path=os.path.join(subdir_path, file),
                                description=descriptions.get(file, default_desc),
                                category=subdir,
                            ))
            return

        for root, _, files in os.walk(base_path):
            for file in sorted(files):
                if file.lower().endswith(VIDEO_EXTENSIONS):
                    self.video_list.append(
                        SVIVideoItem(path=os.path.join(root, file), description="The video", category="unknown"))

    @staticmethod
    def _read_csv(csv_path: str) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Filename")
                    desc = row.get("Video Description")
                    if name and desc is not None:
                        out[name] = desc
        except Exception as exc:
            logger.warning("SVIRawVideoDataset: failed to read csv %s (%s)", csv_path, exc)
        return out

    def __len__(self) -> int:
        return len(self.video_list)

    def _resize(self, image: Image.Image) -> Image.Image:
        return TF.resize(image, (self.height, self.width), interpolation=InterpolationMode.BILINEAR)

    def _sample_frame_indices(self, total_frames: int) -> tuple[list[int], int]:
        stride = random.randint(1, max(1, self.sample_fps))
        cover = stride * self.max_frames

        if total_frames < cover + 1:
            stride = max(total_frames // self.max_frames, 1)
            start = 0
            end = min(stride * self.max_frames, total_frames - 1)
        else:
            max_start = max(0, total_frames - cover - 5)
            start = random.randint(0, max_start) if max_start > 0 else 0
            end = start + cover

        indices = list(range(start, min(end, total_frames), stride))
        while len(indices) < self.max_frames:
            indices.append(indices[-1] if indices else 0)
        return indices[:self.max_frames], stride

    def _random_crop_box(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Pick a random crop preserving the target aspect ratio with mild zoom."""
        target_aspect = self.height / self.width
        jitter = max(int(min(w, h) * self.crop_jitter_frac), 1)
        if w * target_aspect <= h:
            crop_w = random.randint(max(1, w - jitter), w)
            crop_h = int(crop_w * target_aspect)
        else:
            crop_h = random.randint(max(1, h - jitter), h)
            crop_w = int(crop_h / target_aspect)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        max_x = w - crop_w
        max_y = h - crop_h
        x1 = random.randint(0, max_x) if max_x > 0 else 0
        y1 = random.randint(0, max_y) if max_y > 0 else 0
        return x1, y1, x1 + crop_w, y1 + crop_h

    def __getitem__(self, index: int) -> dict:
        index = index % len(self.video_list)
        item = self.video_list[index]
        video_file = item.path

        # Tolerate broken videos by retrying with a random index (upstream behaviour).
        try:
            reader = imageio.get_reader(video_file)
            total_frames = reader.count_frames()
        except (OSError, IOError, ValueError) as exc:
            logger.warning("SVIRawVideoDataset: failed to read %s (%s); retrying random", video_file, exc)
            return self.__getitem__(random.randint(0, len(self.video_list) - 1))

        try:
            indices, _stride = self._sample_frame_indices(total_frames)
            frames: list[Image.Image] = []
            for idx in indices:
                frame = reader.get_data(idx)
                frames.append(Image.fromarray(frame).convert("RGB"))
        finally:
            reader.close()

        if not frames:
            return self.__getitem__(random.randint(0, len(self.video_list) - 1))

        first_ref = [frames[i].copy() for i in range(min(self.num_ref_frames, len(frames)))]
        while len(first_ref) < self.num_ref_frames:
            first_ref.append(first_ref[-1].copy())
        random_ref = frames[random.randint(0, len(frames) - 1)].copy()

        src_w, src_h = first_ref[0].size
        x1, y1, x2, y2 = self._random_crop_box(src_w, src_h)
        first_ref = [f.crop((x1, y1, x2, y2)) for f in first_ref]
        random_ref = random_ref.crop((x1, y1, x2, y2))

        first_ref_tensors = [torch.from_numpy(np.array(self._resize(f))) for f in first_ref]
        random_ref_tensor = torch.from_numpy(np.array(self._resize(random_ref)))

        video = torch.stack(
            [self.frame_process(self._resize(f.crop((x1, y1, x2, y2)))) for f in frames],
            dim=0,
        )
        video = rearrange(video, "T C H W -> C T H W")

        return {
            "text": item.description,
            "video": video,
            "first_ref_frames": first_ref_tensors,
            "random_ref_frame": random_ref_tensor,
            "path": item.path,
        }
