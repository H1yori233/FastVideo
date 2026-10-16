# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo native imports and concise documentation.
"""FLUX3 reference-image preparation and pixel conversion."""

import math

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from fastvideo.models.vaes.flux3 import DOWNSAMPLING_RATIO

MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024


def _ceil16(v: int) -> int:
    return -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Fit a reference inside a 4 MP canvas with 64–4096 pixel sides."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, (canvas[0], canvas[1])
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Return a padded RGB reference and its optional (left, top, right, bottom) crop."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    # Small references need symmetric padding.
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    crop = (left, top, left + image.width, top + image.height)
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode)), crop


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """Fit under 1 MP and center-crop to the 16-pixel grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO, height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def default_size(references: list[Image.Image]) -> tuple[int, int]:
    """1024x1024 for text-to-image; the first (prepared) reference's own size otherwise."""
    return references[0].size if references else (1024, 1024)


def to_tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def to_pil(image: Tensor) -> Image.Image:
    array = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)
