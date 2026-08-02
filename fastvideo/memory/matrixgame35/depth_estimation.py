# SPDX-License-Identifier: Apache-2.0
"""Lazy Depth Anything 3 boundary for Matrix-Game 3.5 Patch Memory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image
import torch

BASE_DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
BASE_DA3_PROCESS_RES = 504
DISTILLED_DA3_PROCESS_RES = 448
DISTILLED_DA3_AUTOCAST_DTYPE = torch.bfloat16


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
        model_ref: str = BASE_DA3_MODEL_ID,
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
                    process_res=BASE_DA3_PROCESS_RES,
                )
        finally:
            if self._cpu_offload:
                self.offload_to_cpu()

        depths = np.asarray(prediction.depth, dtype=np.float32)
        if depths.ndim != 3 or depths.shape[0] != len(pil_frames):
            raise ValueError("Depth Anything 3 must return prediction.depth with shape "
                             f"[{len(pil_frames)}, height, width], got {depths.shape}.")
        return np.ascontiguousarray(depths)


def _load_distilled_depth_anything3(model_ref: str) -> Any:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise ImportError("Matrix-Game 3.5 distilled memory requires the optional Depth Anything 3 package.") from error
    return DepthAnything3.from_pretrained(model_ref)


def _to_distilled_pil_rgb(frame: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"DA3 frames must have shape [H,W,3], got {array.shape}.")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("DA3 frames must contain finite numeric RGB values.")
        array = array * 255.0 if array.size and float(array.max()) <= 1.0 else array
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


class MatrixGame35DistilledDepthAnything3Adapter:
    """Explicit, injectable DA3 boundary for the released 448/BF16 runtime."""

    def __init__(
        self,
        model_ref: str,
        *,
        device: str | torch.device = "cuda",
        estimator: Any | None = None,
        estimator_loader: Callable[[str], Any] | None = None,
        cpu_offload: bool = True,
    ) -> None:
        if not str(model_ref).strip() and estimator is None:
            raise ValueError("An explicit DA3 model path/ref or injected estimator is required.")
        if estimator is not None and estimator_loader is not None:
            raise ValueError("Pass either estimator or estimator_loader, not both.")
        self.model_ref = str(model_ref)
        self.device = torch.device(device)
        self.estimator = estimator
        self.estimator_loader = estimator_loader or _load_distilled_depth_anything3
        self.cpu_offload = bool(cpu_offload)

    def _get_estimator(self) -> Any:
        if self.estimator is None:
            self.estimator = self.estimator_loader(self.model_ref)
        self.estimator.eval()
        return self.estimator

    def estimate_depth(self, frames: Sequence[np.ndarray | Image.Image]) -> np.ndarray:
        """Infer one FP32 metric depth map per frame with DA3 pose disabled."""
        if not frames:
            raise ValueError("DA3 requires at least one RGB frame.")
        estimator = self._get_estimator().to(self.device)
        self.estimator = estimator
        autocast = (torch.autocast(device_type="cuda", dtype=DISTILLED_DA3_AUTOCAST_DTYPE)
                    if self.device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
        try:
            with torch.inference_mode(), autocast:
                prediction = estimator.inference(
                    [_to_distilled_pil_rgb(frame) for frame in frames],
                    use_ray_pose=False,
                    process_res=DISTILLED_DA3_PROCESS_RES,
                )
        finally:
            if self.cpu_offload:
                self.estimator = estimator.to("cpu")
        depths = np.asarray(prediction.depth, dtype=np.float32)
        if depths.ndim != 3 or depths.shape[0] != len(frames):
            raise ValueError("Depth Anything 3 must return prediction.depth shaped "
                             f"[{len(frames)},H,W], got {depths.shape}.")
        return np.ascontiguousarray(depths)


def is_da3_insufficient_non_sky_error(error: BaseException) -> bool:
    """Recognize the released DA3 failure that degrades to empty memory."""
    expected = "Insufficient non-sky pixels for alignment"
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if expected in str(current):
            return True
        current = current.__cause__ or current.__context__
    return expected in repr(error)


__all__ = [
    "BASE_DA3_MODEL_ID",
    "BASE_DA3_PROCESS_RES",
    "DISTILLED_DA3_AUTOCAST_DTYPE",
    "DISTILLED_DA3_PROCESS_RES",
    "MatrixGame35DepthAnything3Adapter",
    "MatrixGame35DistilledDepthAnything3Adapter",
    "is_da3_insufficient_non_sky_error",
]
