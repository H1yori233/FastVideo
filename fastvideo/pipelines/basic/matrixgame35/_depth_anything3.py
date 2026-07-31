# SPDX-License-Identifier: Apache-2.0
"""Lazy Depth Anything 3 boundary for Matrix-Game 3.5 Patch Memory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image
import torch

DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_PROCESS_RES = 504


def _load_depth_anything3(model_ref: str) -> Any:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise ImportError("Matrix-Game 3.5 Patch Memory requires the optional Depth Anything 3 "
                          "package. Install the official depth-anything-3 package to enable it.") from error
    return DepthAnything3.from_pretrained(model_ref)


def _to_pil_rgb(frame: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frames must contain NumPy arrays or PIL images, got {type(frame).__name__}.")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"NumPy frames must have shape [height, width, 3], got {frame.shape}.")
    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.number):
            raise TypeError(f"NumPy frames must have a numeric dtype, got {frame.dtype}.")
        if not np.isfinite(frame).all():
            raise ValueError("NumPy frames must contain only finite values.")
        frame = frame * 255.0 if frame.size and frame.max() <= 1.0 else frame
        frame = frame.astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(frame), mode="RGB")


class MatrixGame35DepthAnything3Adapter:
    """Estimate only the metric depths used by Base-standard Patch Memory."""

    def __init__(
        self,
        model_ref: str = DA3_MODEL_ID,
        *,
        device: str | torch.device = "cuda",
        estimator: Any | None = None,
        estimator_loader: Callable[[str], Any] | None = None,
        cpu_offload: bool = False,
    ) -> None:
        if estimator is not None and estimator_loader is not None:
            raise ValueError("Pass either estimator or estimator_loader, not both.")
        self._model_ref = model_ref
        self._device = torch.device(device)
        self._estimator = estimator
        self._estimator_loader = estimator_loader or _load_depth_anything3
        self._cpu_offload = bool(cpu_offload)

    def _get_estimator(self) -> Any:
        if self._estimator is None:
            self._estimator = self._estimator_loader(self._model_ref)
        self._estimator.eval()
        return self._estimator

    def offload_to_cpu(self) -> None:
        """Park an already-loaded estimator without forcing a lazy load."""
        if self._estimator is not None:
            self._estimator = self._estimator.to("cpu")

    def estimate_depth(self, frames: Sequence[np.ndarray | Image.Image]) -> np.ndarray:
        """Return contiguous CPU FP32 metric depths with shape ``[frames, H, W]``."""
        if not frames:
            raise ValueError("frames must contain at least one RGB image.")
        pil_frames = [_to_pil_rgb(frame) for frame in frames]
        estimator = self._get_estimator()
        self._estimator = estimator.to(self._device)
        try:
            with torch.inference_mode(), torch.autocast(device_type=self._device.type, enabled=False):
                prediction = self._estimator.inference(
                    pil_frames,
                    use_ray_pose=False,
                    process_res=DA3_PROCESS_RES,
                )
        finally:
            if self._cpu_offload:
                self.offload_to_cpu()

        depths = np.asarray(prediction.depth, dtype=np.float32)
        if depths.ndim != 3 or depths.shape[0] != len(pil_frames):
            raise ValueError("Depth Anything 3 must return prediction.depth with shape "
                             f"[{len(pil_frames)}, height, width], got {depths.shape}.")
        return np.ascontiguousarray(depths)
