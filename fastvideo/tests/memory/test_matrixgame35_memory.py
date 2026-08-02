# SPDX-License-Identifier: Apache-2.0
"""Focused CPU contracts for the Matrix-Game 3.5 memory package."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

from fastvideo.memory.matrixgame35.depth_estimation import (
    BASE_DA3_PROCESS_RES,
    DISTILLED_DA3_PROCESS_RES,
    MatrixGame35DepthAnything3Adapter,
    MatrixGame35DistilledDepthAnything3Adapter,
)
from fastvideo.memory.matrixgame35.dynamic_context import (
    MatrixGame35DynamicContextEntry,
    MatrixGame35DynamicContextPool,
)
from fastvideo.memory.matrixgame35.patch_memory import (
    MatrixGame35BasePatchMemory,
    MatrixGame35DistilledPatchMemory,
)


def _pose(*, translation_x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = np.float32(translation_x)
    return pose


def _intrinsic() -> np.ndarray:
    return np.diag((16.0, 16.0, 1.0)).astype(np.float32)


def test_base_memory_append_and_query_contract() -> None:
    memory = MatrixGame35BasePatchMemory()
    memory.append(
        latents=torch.ones(1, 1, 2, 2, dtype=torch.float16),
        w2c=_pose()[None],
        intrinsics=_intrinsic()[None],
        depths=np.ones((1, 2, 2), dtype=np.float64),
    )

    result = memory.query(
        anchor_w2c=_pose(),
        query_w2c=np.repeat(_pose()[None], 84, axis=0),
        query_intrinsics=np.repeat(_intrinsic()[None], 84, axis=0),
    )

    assert memory.latents.dtype == torch.float32
    assert memory.depths.dtype == np.float32
    assert result.latents.shape == (1, 21, 2, 2)
    assert result.valid_mask.all()
    torch.testing.assert_close(result.latents, torch.ones_like(result.latents))


def test_distilled_memory_uses_far_z_fill_policy() -> None:
    memory = MatrixGame35DistilledPatchMemory()
    memory.append(
        latents=torch.stack(
            (torch.ones(8, 8), torch.full((8, 8), 9.0))
        ).unsqueeze(0),
        w2c=np.stack((_pose(translation_x=2.0), _pose())),
        intrinsics=np.repeat(_intrinsic()[None], 2, axis=0),
        depths=np.stack(
            (
                np.full((8, 8), 2.0, dtype=np.float32),
                np.full((8, 8), 4.0, dtype=np.float32),
            )
        ),
    )

    fused, valid = memory.fuse_candidates(
        np.repeat(_pose()[None], 4, axis=0),
        np.repeat(_intrinsic()[None], 4, axis=0),
        ((0, 1, -1, -1, -1), ),
    )

    assert valid.all()
    torch.testing.assert_close(fused, torch.full_like(fused, 9.0))


def test_dynamic_context_selection_and_sink_policy() -> None:
    older = MatrixGame35DynamicContextEntry(
        latent=torch.zeros(1),
        position=1,
        camera_frame=4,
        source_timeline_position=3,
        representative_w2c=_pose(),
    )
    newer = MatrixGame35DynamicContextEntry(
        latent=torch.ones(1),
        position=2,
        camera_frame=8,
        source_timeline_position=7,
        representative_w2c=_pose(translation_x=0.1),
    )
    pool = MatrixGame35DynamicContextPool()
    pool.publish((newer, older))

    assert pool.select(
        _pose()[None],
        exclude_position=-1,
        exclude_camera_frame=-1,
    ) is older

    sink = MatrixGame35DynamicContextPool(original_anchor=older, force_original_anchor=True)
    assert sink.select(
        _pose()[None], chunk_index=0, exclude_position=-1, exclude_camera_frame=-1
    ) is None
    assert sink.select(
        _pose()[None], chunk_index=1, exclude_position=-1, exclude_camera_frame=-1
    ) is older


class _FakeDepthEstimator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def eval(self) -> _FakeDepthEstimator:
        return self

    def to(self, _device: Any) -> _FakeDepthEstimator:
        return self

    def inference(
        self, frames: list[Image.Image], **kwargs: object
    ) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(depth=np.ones((len(frames), 2, 3), dtype=np.float64))


@pytest.mark.parametrize(
    ("adapter_cls", "process_res"),
    (
        (MatrixGame35DepthAnything3Adapter, BASE_DA3_PROCESS_RES),
        (MatrixGame35DistilledDepthAnything3Adapter, DISTILLED_DA3_PROCESS_RES),
    ),
)
def test_depth_adapters_preserve_variant_runtime_contract(adapter_cls, process_res: int) -> None:
    estimator = _FakeDepthEstimator()
    adapter = adapter_cls("local/da3", device="cpu", estimator=estimator)

    depths = adapter.estimate_depth([np.zeros((2, 3, 3), dtype=np.uint8)])

    assert estimator.calls == [{"use_ray_pose": False, "process_res": process_res}]
    assert depths.shape == (1, 2, 3)
    assert depths.dtype == np.float32
    assert depths.flags.c_contiguous
