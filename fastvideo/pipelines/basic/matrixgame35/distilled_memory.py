# SPDX-License-Identifier: Apache-2.0
"""Released online memory and dynamic visual context for distilled profiles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch

from fastvideo.pipelines.basic.matrixgame35.patch_memory import (
    MatrixGame35BasePatchMemory,
    _nearest_resize_depth,
)

DA3_PROCESS_RES = 448
DA3_AUTOCAST_DTYPE = torch.bfloat16
MEMORY_CANDIDATES_PER_GROUP = 5
MEMORY_COVERAGE_GRID_DOWNSAMPLE = 4
MEMORY_COVERAGE_POOL_STRIDE = 2
MEMORY_FILL_RATIO = 0.95
RGB_SUBFRAMES_PER_LATENT = 4


def _load_da3(model_ref: str) -> Any:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise ImportError("Matrix-Game 3.5 distilled memory requires the optional Depth Anything 3 package.") from error
    return DepthAnything3.from_pretrained(model_ref)


def _to_pil(frame: np.ndarray | Image.Image) -> Image.Image:
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
        self.estimator_loader = estimator_loader or _load_da3
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
        autocast = (torch.autocast(device_type="cuda", dtype=DA3_AUTOCAST_DTYPE)
                    if self.device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
        try:
            with torch.inference_mode(), autocast:
                prediction = estimator.inference(
                    [_to_pil(frame) for frame in frames],
                    use_ray_pose=False,
                    process_res=DA3_PROCESS_RES,
                )
        finally:
            if self.cpu_offload:
                self.estimator = estimator.to("cpu")
        depths = np.asarray(prediction.depth, dtype=np.float32)
        if depths.ndim != 3 or depths.shape[0] != len(frames):
            raise ValueError("Depth Anything 3 must return prediction.depth shaped "
                             f"[{len(frames)},H,W], got {depths.shape}.")
        return np.ascontiguousarray(depths)


@dataclass(frozen=True)
class MatrixGame35DistilledMemoryResult:
    """One three-latent mosaic and its geometry-selection evidence."""

    latents: torch.Tensor
    valid_mask: torch.Tensor
    candidate_frame_ids: tuple[tuple[int, ...], ...]
    aligned_query_w2c: np.ndarray


class MatrixGame35DistilledPatchMemory(MatrixGame35BasePatchMemory):
    """C0 + generated append-only memory with released STANDARD policies."""

    @staticmethod
    def _align_query(anchor_w2c: np.ndarray, query_w2c: np.ndarray, target_w2c: np.ndarray) -> np.ndarray:
        anchor = np.asarray(anchor_w2c, dtype=np.float32)
        query = np.asarray(query_w2c, dtype=np.float32)
        target = np.asarray(target_w2c, dtype=np.float32)
        if anchor.shape != (4, 4) or query.ndim != 3 or query.shape[1:] != (4, 4):
            raise ValueError("anchor_w2c/query_w2c must have shapes [4,4] and [Q,4,4].")
        trajectory = np.concatenate((anchor[None], query), axis=0)
        aligned = trajectory @ (np.linalg.inv(trajectory[0]) @ target)
        return np.ascontiguousarray(aligned[1:].astype(np.float32))

    def select_candidate_frame_ids(
        self,
        query_w2c: np.ndarray,
        query_intrinsics: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        """Select five frames/group by greedy union coverage over a stride-2 pool."""
        query = np.asarray(query_w2c, dtype=np.float64)
        query_K = np.asarray(query_intrinsics, dtype=np.float64)
        if query.ndim != 3 or query.shape[1:] != (4, 4) or query.shape[0] % RGB_SUBFRAMES_PER_LATENT:
            raise ValueError("distilled memory query poses must have shape [4*k,4,4].")
        if query_K.shape != (query.shape[0], 3, 3):
            raise ValueError(f"query intrinsics must have shape [{query.shape[0]},3,3].")

        height, width = map(int, self.latents.shape[-2:])
        downsample = MEMORY_COVERAGE_GRID_DOWNSAMPLE
        coarse_height, coarse_width = height // downsample, width // downsample
        if coarse_height <= 0 or coarse_width <= 0:
            raise ValueError("latent dimensions must be at least the coverage downsample factor.")
        pool = np.arange(0, self.num_frames, MEMORY_COVERAGE_POOL_STRIDE)
        memory_count = len(pool)
        group_count = query.shape[0] // RGB_SUBFRAMES_PER_LATENT

        depth_grid = np.stack(
            [_nearest_resize_depth(self.depths[int(frame_id)], coarse_height, coarse_width) for frame_id in pool],
            axis=0,
        ).astype(np.float64)
        rows, columns = np.meshgrid(np.arange(coarse_height), np.arange(coarse_width), indexing="ij")
        stride = 16 * downsample
        pixels = np.stack(
            (
                (columns.reshape(-1) * stride).astype(np.float64),
                (rows.reshape(-1) * stride).astype(np.float64),
                np.ones(coarse_height * coarse_width, dtype=np.float64),
            ),
            axis=0,
        )
        point_count = pixels.shape[1]
        depth_flat = depth_grid.reshape(memory_count, point_count)
        memory_w2c = self.w2c.astype(np.float64)[pool]
        inverse_K = np.linalg.inv(self.intrinsics.astype(np.float64)[pool])
        rays = np.einsum("mij,jp->mip", inverse_K, pixels)
        camera_points = rays * depth_flat[:, None, :] - memory_w2c[:, :3, 3, None]
        world_points = np.einsum("mji,mjp->mip", memory_w2c[:, :3, :3], camera_points)

        query_indices = [group * RGB_SUBFRAMES_PER_LATENT + 3 for group in range(group_count)]
        query_rotation = query[query_indices, :3, :3]
        query_translation = query[query_indices, :3, 3]
        projected_camera = (np.einsum("gij,mjp->gmip", query_rotation, world_points) +
                            query_translation[:, None, :, None])
        projected = np.einsum("gij,gmjp->gmip", query_K[query_indices], projected_camera)
        depth = projected[:, :, 2]
        valid = (np.isfinite(depth) & (depth > 1e-2) & np.isfinite(depth_flat)[None] & (depth_flat[None] > 1e-3))
        denominator = np.where(valid, depth, 1.0)
        full_rows = np.rint(projected[:, :, 1] / denominator / 16.0).astype(np.int64)
        full_columns = np.rint(projected[:, :, 0] / denominator / 16.0).astype(np.int64)
        coarse_rows = full_rows // downsample
        coarse_columns = full_columns // downsample
        in_bounds = (valid
                     & (coarse_rows >= 0)
                     & (coarse_rows < coarse_height)
                     & (coarse_columns >= 0)
                     & (coarse_columns < coarse_width))
        masks = np.zeros((group_count, memory_count, coarse_height, coarse_width), dtype=bool)
        group_ids, memory_ids, point_ids = np.where(in_bounds)
        masks[
            group_ids,
            memory_ids,
            coarse_rows[group_ids, memory_ids, point_ids],
            coarse_columns[group_ids, memory_ids, point_ids],
        ] = True

        selected: list[tuple[int, ...]] = []
        flat_masks = masks.reshape(group_count, memory_count, -1)
        areas = flat_masks.sum(axis=2)
        for group in range(group_count):
            union = np.zeros(flat_masks.shape[-1], dtype=bool)
            available = np.ones(memory_count, dtype=bool)
            picked: list[int] = []
            for _ in range(MEMORY_CANDIDATES_PER_GROUP):
                gain = (flat_masks[group] & ~union).sum(axis=1)
                gain[~available] = -1
                best = int(gain.argmax())
                if gain[best] <= 0:
                    break
                union |= flat_masks[group, best]
                picked.append(int(pool[best]))
                available[best] = False
            if len(picked) < MEMORY_CANDIDATES_PER_GROUP:
                remaining_area = areas[group].copy()
                remaining_area[~available] = -1
                for best in np.argsort(-remaining_area):
                    if available[best] and remaining_area[best] > 0:
                        picked.append(int(pool[best]))
                        available[best] = False
                    if len(picked) == MEMORY_CANDIDATES_PER_GROUP:
                        break
            if not picked:
                picked = [-1]
            picked.extend([picked[-1]] * (MEMORY_CANDIDATES_PER_GROUP - len(picked)))
            selected.append(tuple(picked))
        return tuple(selected)

    def fuse_candidates(
        self,
        query_w2c: np.ndarray,
        query_intrinsics: np.ndarray,
        candidate_frame_ids: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest splat per candidate, then far-z fill-stop across candidates."""
        query = np.asarray(query_w2c, dtype=np.float32)
        query_K = np.asarray(query_intrinsics, dtype=np.float32)
        expected_groups = query.shape[0] // RGB_SUBFRAMES_PER_LATENT
        if len(candidate_frame_ids) != expected_groups:
            raise ValueError(f"expected {expected_groups} candidate groups, got {len(candidate_frame_ids)}.")

        output_latents: list[torch.Tensor] = []
        output_masks: list[torch.Tensor] = []
        for group, frame_ids in enumerate(candidate_frame_ids):
            query_index = group * RGB_SUBFRAMES_PER_LATENT + 3
            fused = torch.zeros_like(self.latents[:, :1])
            filled = torch.zeros(self.latents.shape[-2:], dtype=torch.bool)
            best_depth = torch.full(self.latents.shape[-2:], float("-inf"), dtype=torch.float32)
            unique_ids: list[int] = []
            for frame_id in frame_ids:
                frame_id = int(frame_id)
                if frame_id >= 0 and frame_id not in unique_ids:
                    unique_ids.append(frame_id)
            if not unique_ids:
                unique_ids = [0]
            for frame_id in unique_ids:
                if frame_id >= self.num_frames:
                    raise ValueError(f"candidate frame {frame_id} is outside the {self.num_frames}-frame bank.")
                splat, valid, depth = self._splat_candidate(
                    frame_id,
                    query_w2c=query[query_index],
                    query_K=query_K[query_index],
                )
                wins = valid & (depth > best_depth)
                fused = torch.where(wins[None, None], splat, fused)
                filled |= wins
                best_depth = torch.where(wins, depth, best_depth)
                if float(filled.float().mean()) >= MEMORY_FILL_RATIO:
                    break
            output_latents.append(fused)
            output_masks.append(filled)
        return torch.cat(output_latents, dim=1), torch.stack(output_masks)

    def query(
        self,
        *,
        anchor_w2c: np.ndarray,
        query_w2c: np.ndarray,
        query_intrinsics: np.ndarray,
    ) -> MatrixGame35DistilledMemoryResult:
        """Align, select, and fuse the current three-latent mosaic."""
        aligned = self._align_query(anchor_w2c, query_w2c, self.w2c[-1])
        candidates = self.select_candidate_frame_ids(aligned, query_intrinsics)
        latents, valid_mask = self.fuse_candidates(aligned, query_intrinsics, candidates)
        return MatrixGame35DistilledMemoryResult(latents, valid_mask, candidates, aligned)


@dataclass(frozen=True)
class MatrixGame35DynamicContextEntry:
    """One generated context latent with global temporal/camera provenance."""

    latent: torch.Tensor
    position: int
    camera_frame: int
    source_timeline_position: int
    representative_w2c: np.ndarray
    source: str = "generated"


def _camera_center(w2c: np.ndarray) -> np.ndarray:
    matrix = np.asarray(w2c, dtype=np.float64)
    return -(matrix[:3, :3].T @ matrix[:3, 3])


def dynamic_context_pose_score(entry_w2c: np.ndarray, target_w2c: np.ndarray) -> float:
    """Match upstream translation-mean plus rotation-radian pose distance."""
    entry = np.asarray(entry_w2c, dtype=np.float64)
    targets = np.asarray(target_w2c, dtype=np.float64)
    if targets.ndim == 2:
        targets = targets[None]
    centers = np.stack([_camera_center(target) for target in targets])
    translation = float(np.linalg.norm(centers - _camera_center(entry)[None], axis=-1).mean())
    rotations = []
    for target in targets:
        relative = target[:3, :3] @ entry[:3, :3].T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        rotations.append(float(np.arccos(cosine)))
    return translation + float(np.mean(rotations))


class MatrixGame35DynamicContextPool:
    """Published contexts with an optional original-anchor sink policy."""

    def __init__(
        self,
        pose_pool_size: int = 5,
        *,
        original_anchor: MatrixGame35DynamicContextEntry | None = None,
        force_original_anchor: bool = False,
    ) -> None:
        if int(pose_pool_size) <= 0:
            raise ValueError("pose_pool_size must be positive.")
        self.pose_pool_size = int(pose_pool_size)
        self.entries: list[MatrixGame35DynamicContextEntry] = []
        self.original_anchor = original_anchor
        self.force_original_anchor = bool(force_original_anchor)

    def publish(self, entries: Sequence[MatrixGame35DynamicContextEntry]) -> None:
        self.entries.extend(entries)

    def select(
        self,
        target_w2c: np.ndarray,
        *,
        chunk_index: int = 0,
        exclude_position: int,
        exclude_camera_frame: int,
    ) -> MatrixGame35DynamicContextEntry | None:
        if self.force_original_anchor:
            return None if int(chunk_index) == 0 else self.original_anchor
        candidates = [
            entry for entry in self.entries
            if entry.position != int(exclude_position) and entry.camera_frame != int(exclude_camera_frame)
        ]
        if not candidates:
            return None
        scored = [(
            dynamic_context_pose_score(entry.representative_w2c, target_w2c),
            -entry.position,
            stable_index,
            entry,
        ) for stable_index, entry in enumerate(candidates)]
        near = sorted(scored, key=lambda item: item[:3])[:self.pose_pool_size]
        return min(
            near,
            key=lambda item: (
                item[3].source_timeline_position,
                item[0],
                item[3].position,
                item[2],
            ),
        )[3]


def is_da3_insufficient_non_sky_error(error: BaseException) -> bool:
    """Recognize the one released DA3 failure that degrades to empty memory."""
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
    "DA3_AUTOCAST_DTYPE",
    "DA3_PROCESS_RES",
    "MatrixGame35DistilledDepthAnything3Adapter",
    "MatrixGame35DistilledMemoryResult",
    "MatrixGame35DistilledPatchMemory",
    "MatrixGame35DynamicContextEntry",
    "MatrixGame35DynamicContextPool",
    "dynamic_context_pose_score",
    "is_da3_insufficient_non_sky_error",
]
