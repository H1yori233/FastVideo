# SPDX-License-Identifier: Apache-2.0
"""Base-standard Patch Memory for Matrix-Game 3.5."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from fastvideo.memory.matrixgame35.constants import (
    RGB_FRAMES_PER_BLOCK,
    RGB_SUBFRAMES_PER_LATENT,
)

LATENT_STRIDE = 16
CANDIDATES_PER_QUERY_GROUP = 5
POSE_NMS_DISTANCE = 0.1
CANDIDATE_POOL_MULTIPLIER = 2.0
FILL_RATIO_THRESHOLD = 0.95
DISTILLED_CANDIDATES_PER_QUERY_GROUP = 5
DISTILLED_COVERAGE_GRID_DOWNSAMPLE = 4
DISTILLED_COVERAGE_POOL_STRIDE = 2
DISTILLED_FILL_RATIO_THRESHOLD = 0.95


class MatrixGame35DepthAdapter(Protocol):
    """Optional boundary for DA3-compatible metric-depth adapters."""

    def estimate_depth(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Return one metric FP32 depth map per RGB frame."""


@dataclass(frozen=True)
class MatrixGame35PatchMemoryResult:
    """Mosaic latents and selection evidence for one 84-frame Base block."""

    latents: torch.Tensor
    valid_mask: torch.Tensor
    candidate_frame_ids: tuple[tuple[int, ...], ...]
    aligned_query_w2c: np.ndarray


@dataclass(frozen=True)
class MatrixGame35DistilledMemoryResult:
    """One three-latent mosaic and its geometry-selection evidence."""

    latents: torch.Tensor
    valid_mask: torch.Tensor
    candidate_frame_ids: tuple[tuple[int, ...], ...]
    aligned_query_w2c: np.ndarray


def _as_float32_array(value: np.ndarray, *, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape [frames, {', '.join(map(str, shape_tail))}], got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame.")
    return np.ascontiguousarray(array)


def _camera_centers(w2c: np.ndarray) -> np.ndarray:
    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), translation)


def _nearest_resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth maps must be two-dimensional, got {depth.shape}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"latent dimensions must be positive, got {height}x{width}.")
    source_height, source_width = depth.shape
    rows = np.floor(np.arange(height, dtype=np.float64) * source_height / height).astype(np.int64)
    columns = np.floor(np.arange(width, dtype=np.float64) * source_width / width).astype(np.int64)
    return np.ascontiguousarray(depth[rows[:, None], columns[None, :]])


def _reproject_memory_grid(
    depth: np.ndarray,
    memory_K: np.ndarray,
    memory_w2c: np.ndarray,
    query_K: np.ndarray,
    query_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels_hw = np.stack((rows, columns), axis=-1) * LATENT_STRIDE
    pixels_homogeneous = np.stack(
        (pixels_hw[..., 1], pixels_hw[..., 0], np.ones((height, width), dtype=np.float32)),
        axis=-1,
    )

    rays = np.einsum("ij,...j->...i", np.linalg.inv(memory_K), pixels_homogeneous)
    points_memory = rays * depth[..., None]
    memory_rotation = memory_w2c[:3, :3]
    points_world = np.einsum(
        "ij,...j->...i",
        memory_rotation.T,
        points_memory - memory_w2c[:3, 3],
    )
    points_query = np.einsum("ij,...j->...i", query_w2c[:3, :3], points_world) + query_w2c[:3, 3]
    query_depth = points_query[..., 2]
    valid = np.isfinite(query_depth) & (query_depth > 1e-2)
    denominator = np.where(valid, query_depth, 1.0)
    normalized = np.stack(
        (
            np.divide(points_query[..., 0], denominator),
            np.divide(points_query[..., 1], denominator),
            np.ones((height, width), dtype=np.float32),
        ),
        axis=-1,
    )
    projected = np.einsum("ij,...j->...i", query_K, normalized)
    projected_hw = np.stack((projected[..., 1], projected[..., 0]), axis=-1)
    projected_hw = np.where(valid[..., None], projected_hw, -10000)
    return projected_hw, np.asarray(query_depth, dtype=np.float32)


class MatrixGame35BasePatchMemory:
    """Append-only CPU FP32 Patch Memory for the released Base variants."""

    def __init__(self) -> None:
        self._latents: torch.Tensor | None = None
        self._w2c: np.ndarray | None = None
        self._intrinsics: np.ndarray | None = None
        self._depths: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return 0 if self._latents is None else int(self._latents.shape[1])

    @property
    def latents(self) -> torch.Tensor:
        if self._latents is None:
            raise RuntimeError("Patch Memory is empty.")
        return self._latents

    @property
    def w2c(self) -> np.ndarray:
        if self._w2c is None:
            raise RuntimeError("Patch Memory is empty.")
        return self._w2c

    @property
    def intrinsics(self) -> np.ndarray:
        if self._intrinsics is None:
            raise RuntimeError("Patch Memory is empty.")
        return self._intrinsics

    @property
    def depths(self) -> np.ndarray:
        if self._depths is None:
            raise RuntimeError("Patch Memory is empty.")
        return self._depths

    def append(
        self,
        *,
        latents: torch.Tensor,
        w2c: np.ndarray,
        intrinsics: np.ndarray,
        depths: np.ndarray | None = None,
        frames: Sequence[np.ndarray] | None = None,
        depth_adapter: MatrixGame35DepthAdapter | None = None,
    ) -> None:
        """Append one frame-aligned batch without retaining RGB frames."""
        if latents.ndim != 4 or latents.shape[1] == 0:
            raise ValueError(f"latents must have shape [channels, frames, height, width], got {tuple(latents.shape)}.")
        frame_count = int(latents.shape[1])
        w2c_array = _as_float32_array(w2c, name="w2c", shape_tail=(4, 4))
        intrinsics_array = _as_float32_array(intrinsics, name="intrinsics", shape_tail=(3, 3))
        if w2c_array.shape[0] != frame_count or intrinsics_array.shape[0] != frame_count:
            raise ValueError("latents, w2c, and intrinsics must contain the same number of frames.")

        if depths is None:
            if depth_adapter is None or frames is None:
                raise ValueError("depths or both frames and a depth_adapter are required.")
            if len(frames) != frame_count:
                raise ValueError(f"frames must contain {frame_count} items, got {len(frames)}.")
            depths = depth_adapter.estimate_depth(frames)
        depth_array = _as_float32_array(depths, name="depths", shape_tail=np.asarray(depths).shape[1:])
        if depth_array.ndim != 3 or depth_array.shape[0] != frame_count:
            raise ValueError(f"depths must have shape [{frame_count}, height, width], got {depth_array.shape}.")

        latent_array = latents.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if self._latents is None:
            self._latents = latent_array
            self._w2c = w2c_array.copy()
            self._intrinsics = intrinsics_array.copy()
            self._depths = depth_array.copy()
            return

        if latent_array.shape[0] != self._latents.shape[0] or latent_array.shape[2:] != self._latents.shape[2:]:
            raise ValueError("appended latent channels and spatial dimensions must match the bank, "
                             f"got {tuple(latent_array.shape)} after {tuple(self._latents.shape)}.")
        if depth_array.shape[1:] != self.depths.shape[1:]:
            raise ValueError(
                f"appended depth dimensions must match the bank, got {depth_array.shape[1:]} after {self.depths.shape[1:]}."
            )
        self._latents = torch.cat((self._latents, latent_array), dim=1)
        self._w2c = np.concatenate((self.w2c, w2c_array), axis=0)
        self._intrinsics = np.concatenate((self.intrinsics, intrinsics_array), axis=0)
        self._depths = np.concatenate((self.depths, depth_array), axis=0)

    def align_query_trajectory(self, anchor_w2c: np.ndarray, query_w2c: np.ndarray) -> np.ndarray:
        """Align ``anchor + 84 query poses`` to the latest memory pose."""
        anchor = np.asarray(anchor_w2c, dtype=np.float32)
        query = _as_float32_array(query_w2c, name="query_w2c", shape_tail=(4, 4))
        if anchor.shape != (4, 4):
            raise ValueError(f"anchor_w2c must have shape [4, 4], got {anchor.shape}.")
        if query.shape[0] != RGB_FRAMES_PER_BLOCK:
            raise ValueError(
                f"Base Patch Memory requires exactly {RGB_FRAMES_PER_BLOCK} query poses, got {query.shape[0]}.")
        trajectory = np.concatenate((anchor[None], query), axis=0)
        alignment = np.linalg.inv(trajectory[0]) @ self.w2c[-1]
        return np.ascontiguousarray((trajectory @ alignment)[1:].astype(np.float32))

    def select_candidate_frame_ids(self, aligned_query_w2c: np.ndarray) -> tuple[tuple[int, ...], ...]:
        """Select five pose-nearest candidates per fourth-pose query group."""
        query = _as_float32_array(aligned_query_w2c, name="aligned_query_w2c", shape_tail=(4, 4))
        if query.shape[0] != RGB_FRAMES_PER_BLOCK:
            raise ValueError(
                f"Base Patch Memory requires exactly {RGB_FRAMES_PER_BLOCK} aligned query poses, got {query.shape[0]}.")
        memory_centers = _camera_centers(self.w2c.astype(np.float64))
        nms_centers = _camera_centers(self.w2c)
        memory_directions = self.w2c.astype(np.float64)[:, 2, :3]
        memory_directions /= np.linalg.norm(memory_directions, axis=1, keepdims=True) + 1e-12

        max_candidates = CANDIDATES_PER_QUERY_GROUP
        pool_topk = int(round(max(max_candidates, min(10, self.num_frames)) * CANDIDATE_POOL_MULTIPLIER))
        pool_topk = min(max(pool_topk, max_candidates), self.num_frames)
        groups: list[tuple[int, ...]] = []
        for query_index in range(RGB_SUBFRAMES_PER_LATENT - 1, RGB_FRAMES_PER_BLOCK, RGB_SUBFRAMES_PER_LATENT):
            query_pose = query[query_index].astype(np.float64)
            query_direction = query_pose[2, :3]
            query_direction /= np.linalg.norm(query_direction) + 1e-12
            angles = np.degrees(np.arccos(np.clip(memory_directions @ query_direction, -1.0, 1.0)))
            angle_pool = np.argsort(angles)[:pool_topk]
            query_rotation = query_pose[:3, :3]
            query_center = -query_rotation.T @ query_pose[:3, 3]
            distances = np.linalg.norm(memory_centers[angle_pool] - query_center[None], axis=1)
            ranked = angle_pool[np.argsort(distances)].tolist()

            picked: list[int] = []
            used: set[int] = set()
            for frame_id in ranked:
                if any(
                        np.linalg.norm(nms_centers[frame_id] - nms_centers[other]) < POSE_NMS_DISTANCE
                        for other in picked):
                    continue
                picked.append(int(frame_id))
                used.add(int(frame_id))
                if len(picked) == max_candidates:
                    break
            if len(picked) < max_candidates:
                for frame_id in ranked:
                    if int(frame_id) in used:
                        continue
                    picked.append(int(frame_id))
                    if len(picked) == max_candidates:
                        break
            picked.extend([-1] * (max_candidates - len(picked)))
            groups.append(tuple(picked))
        return tuple(groups)

    def _splat_candidate(
        self,
        frame_id: int,
        *,
        query_w2c: np.ndarray,
        query_K: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels, _, height, width = self.latents.shape
        depth = _nearest_resize_depth(self.depths[frame_id], height, width)
        projected_hw, query_depth = _reproject_memory_grid(
            depth,
            self.intrinsics[frame_id],
            self.w2c[frame_id],
            query_K,
            query_w2c,
        )
        projected_latent = projected_hw / float(LATENT_STRIDE)
        query_rows = np.rint(projected_latent[..., 0]).astype(np.int64)
        query_columns = np.rint(projected_latent[..., 1]).astype(np.int64)
        valid_np = (np.isfinite(projected_latent).all(axis=-1)
                    & np.isfinite(query_depth)
                    & (query_rows >= 0)
                    & (query_rows < height)
                    & (query_columns >= 0)
                    & (query_columns < width)
                    & (query_depth > 0.0))

        output = torch.zeros((channels, 1, height, width), dtype=torch.float32)
        valid = torch.zeros((height * width, ), dtype=torch.bool)
        z_buffer = np.full((height * width, ), np.inf, dtype=np.float32)
        if not valid_np.any():
            return output, valid.view(height, width), torch.from_numpy(z_buffer.reshape(height, width))

        target = (query_rows * width + query_columns)[valid_np]
        depths = query_depth[valid_np].astype(np.float32)
        np.minimum.at(z_buffer, target, depths)
        winner = depths == z_buffer[target]
        target = target[winner]
        source = np.arange(height * width, dtype=np.int64).reshape(height, width)[valid_np][winner]
        target_t = torch.from_numpy(target)
        source_t = torch.from_numpy(source)
        output.view(channels, -1)[:, target_t] = self.latents[:, frame_id].reshape(channels, -1)[:, source_t]
        valid[target_t] = True
        return output, valid.view(height, width), torch.from_numpy(z_buffer.reshape(height, width))

    def fuse_candidates(
        self,
        aligned_query_w2c: np.ndarray,
        candidate_frame_ids: Sequence[Sequence[int]],
        *,
        query_intrinsics: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-warp and near-z-buffer the selected candidates."""
        query = _as_float32_array(aligned_query_w2c, name="aligned_query_w2c", shape_tail=(4, 4))
        expected_groups = RGB_FRAMES_PER_BLOCK // RGB_SUBFRAMES_PER_LATENT
        if query.shape[0] != RGB_FRAMES_PER_BLOCK or len(candidate_frame_ids) != expected_groups:
            raise ValueError(f"expected {RGB_FRAMES_PER_BLOCK} query poses and {expected_groups} candidate groups.")
        if query_intrinsics is None:
            query_K = np.repeat(self.intrinsics[-1:], RGB_FRAMES_PER_BLOCK, axis=0)
        else:
            query_K = _as_float32_array(query_intrinsics, name="query_intrinsics", shape_tail=(3, 3))
            if query_K.shape[0] != RGB_FRAMES_PER_BLOCK:
                raise ValueError(
                    f"query_intrinsics must contain {RGB_FRAMES_PER_BLOCK} frames, got {query_K.shape[0]}.")

        group_latents: list[torch.Tensor] = []
        group_masks: list[torch.Tensor] = []
        for group_index, frame_ids in enumerate(candidate_frame_ids):
            query_index = group_index * RGB_SUBFRAMES_PER_LATENT + (RGB_SUBFRAMES_PER_LATENT - 1)
            unique_ids: list[int] = []
            for frame_id in frame_ids:
                frame_id = int(frame_id)
                if frame_id < 0 or frame_id in unique_ids:
                    continue
                if frame_id >= self.num_frames:
                    raise ValueError(f"candidate frame {frame_id} is outside the {self.num_frames}-frame bank.")
                unique_ids.append(frame_id)
            if not unique_ids:
                unique_ids = [0]

            fused = torch.zeros_like(self.latents[:, :1])
            filled = torch.zeros(self.latents.shape[-2:], dtype=torch.bool)
            best_depth = torch.full(self.latents.shape[-2:], float("inf"), dtype=torch.float32)
            for frame_id in unique_ids:
                splat, valid, depth = self._splat_candidate(
                    frame_id,
                    query_w2c=query[query_index],
                    query_K=query_K[query_index],
                )
                wins = valid & (depth < best_depth)
                fused = torch.where(wins[None, None], splat, fused)
                filled |= wins
                best_depth = torch.where(wins, depth, best_depth)
                if float(filled.float().mean()) >= FILL_RATIO_THRESHOLD:
                    break
            group_latents.append(fused)
            group_masks.append(filled)
        return torch.cat(group_latents, dim=1), torch.stack(group_masks)

    def query(
        self,
        *,
        anchor_w2c: np.ndarray,
        query_w2c: np.ndarray,
        query_intrinsics: np.ndarray | None = None,
    ) -> MatrixGame35PatchMemoryResult:
        """Materialize the standard Base mosaic for one generated block."""
        aligned = self.align_query_trajectory(anchor_w2c, query_w2c)
        candidate_ids = self.select_candidate_frame_ids(aligned)
        latents, valid_mask = self.fuse_candidates(
            aligned,
            candidate_ids,
            query_intrinsics=query_intrinsics,
        )
        return MatrixGame35PatchMemoryResult(
            latents=latents,
            valid_mask=valid_mask,
            candidate_frame_ids=candidate_ids,
            aligned_query_w2c=aligned,
        )


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
        downsample = DISTILLED_COVERAGE_GRID_DOWNSAMPLE
        coarse_height, coarse_width = height // downsample, width // downsample
        if coarse_height <= 0 or coarse_width <= 0:
            raise ValueError("latent dimensions must be at least the coverage downsample factor.")
        pool = np.arange(0, self.num_frames, DISTILLED_COVERAGE_POOL_STRIDE)
        memory_count = len(pool)
        group_count = query.shape[0] // RGB_SUBFRAMES_PER_LATENT

        depth_grid = np.stack(
            [_nearest_resize_depth(self.depths[int(frame_id)], coarse_height, coarse_width) for frame_id in pool],
            axis=0,
        ).astype(np.float64)
        rows, columns = np.meshgrid(np.arange(coarse_height), np.arange(coarse_width), indexing="ij")
        stride = LATENT_STRIDE * downsample
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

        query_indices = [
            group * RGB_SUBFRAMES_PER_LATENT + (RGB_SUBFRAMES_PER_LATENT - 1) for group in range(group_count)
        ]
        query_rotation = query[query_indices, :3, :3]
        query_translation = query[query_indices, :3, 3]
        projected_camera = (np.einsum("gij,mjp->gmip", query_rotation, world_points) +
                            query_translation[:, None, :, None])
        projected = np.einsum("gij,gmjp->gmip", query_K[query_indices], projected_camera)
        depth = projected[:, :, 2]
        valid = (np.isfinite(depth) & (depth > 1e-2) & np.isfinite(depth_flat)[None] & (depth_flat[None] > 1e-3))
        denominator = np.where(valid, depth, 1.0)
        full_rows = np.rint(projected[:, :, 1] / denominator / LATENT_STRIDE).astype(np.int64)
        full_columns = np.rint(projected[:, :, 0] / denominator / LATENT_STRIDE).astype(np.int64)
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
            for _ in range(DISTILLED_CANDIDATES_PER_QUERY_GROUP):
                gain = (flat_masks[group] & ~union).sum(axis=1)
                gain[~available] = -1
                best = int(gain.argmax())
                if gain[best] <= 0:
                    break
                union |= flat_masks[group, best]
                picked.append(int(pool[best]))
                available[best] = False
            if len(picked) < DISTILLED_CANDIDATES_PER_QUERY_GROUP:
                remaining_area = areas[group].copy()
                remaining_area[~available] = -1
                for best in np.argsort(-remaining_area):
                    if available[best] and remaining_area[best] > 0:
                        picked.append(int(pool[best]))
                        available[best] = False
                    if len(picked) == DISTILLED_CANDIDATES_PER_QUERY_GROUP:
                        break
            if not picked:
                picked = [-1]
            picked.extend([picked[-1]] * (DISTILLED_CANDIDATES_PER_QUERY_GROUP - len(picked)))
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
            query_index = group * RGB_SUBFRAMES_PER_LATENT + (RGB_SUBFRAMES_PER_LATENT - 1)
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
                if float(filled.float().mean()) >= DISTILLED_FILL_RATIO_THRESHOLD:
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


__all__ = [
    "MatrixGame35BasePatchMemory",
    "MatrixGame35DepthAdapter",
    "MatrixGame35DistilledMemoryResult",
    "MatrixGame35DistilledPatchMemory",
    "MatrixGame35PatchMemoryResult",
]
