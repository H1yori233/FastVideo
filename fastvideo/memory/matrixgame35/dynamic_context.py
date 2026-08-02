# SPDX-License-Identifier: Apache-2.0
"""Dynamic visual-context selection for Matrix-Game 3.5 rollout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


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
    """Return translation-mean plus rotation-radian pose distance."""
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


__all__ = [
    "MatrixGame35DynamicContextEntry",
    "MatrixGame35DynamicContextPool",
    "dynamic_context_pose_score",
]
