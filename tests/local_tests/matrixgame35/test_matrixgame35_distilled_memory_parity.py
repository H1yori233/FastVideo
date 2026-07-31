# SPDX-License-Identifier: Apache-2.0
"""CPU differential parity for Matrix-Game 3.5 distilled Patch Memory.

Coverage scope: implementation_subcomponent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.distilled_standard_memory import (
    MatrixGame35DistilledPatchMemory,
)
from tests.local_tests.matrixgame35._upstream_patch_memory import (
    load_upstream_patch_memory,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(
    os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5")
)
PARITY_SCOPE = "implementation_subcomponent"


def _pose(*, translation_x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = np.float32(translation_x)
    return pose


def _intrinsic(focal_length: float = 16.0) -> np.ndarray:
    return np.array(
        (
            (focal_length, 0.0, 0.0),
            (0.0, focal_length, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def _load_upstream():
    source = OFFICIAL_REF_DIR / "frustum" / "frustum_handler.py"
    if not source.is_file():
        pytest.skip(
            "Pinned Matrix-Game 3.5 source is unavailable; set "
            "MATRIXGAME35_OFFICIAL_REF_DIR to the prepared reference clone."
        )
    return load_upstream_patch_memory(OFFICIAL_REF_DIR)


def _append_memory(
    *,
    latents: torch.Tensor,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    depths: np.ndarray,
) -> MatrixGame35DistilledPatchMemory:
    memory = MatrixGame35DistilledPatchMemory()
    memory.append(
        latents=latents,
        w2c=w2c,
        intrinsics=intrinsics,
        depths=depths,
    )
    return memory


def _upstream_handler(module, memory: MatrixGame35DistilledPatchMemory):
    height, width = memory.latents.shape[-2:]
    handler = module.FrustumHandler(
        memory.intrinsics[-1],
        image_size=(height, width),
        latent_stride=16,
        use_gpu=False,
    )
    handler._Kpix = memory.intrinsics[-1].copy()
    handler._Kpixs = [intrinsic.copy() for intrinsic in memory.intrinsics]
    return handler


def test_coverage_selector_matches_pinned_full_pool_greedy_policy() -> None:
    upstream = _load_upstream()
    height = width = 16
    intrinsic = _intrinsic()

    # Only even frames enter the released stride-2 pool. Their shifts yield
    # unequal individual coverage, then complementary union coverage, so the
    # greedy order is observably different from chronological ordering.
    translations = (
        -8.0,
        100.0,
        -4.0,
        100.0,
        4.0,
        100.0,
        8.0,
        100.0,
        -12.0,
        100.0,
        12.0,
        100.0,
    )
    w2c = np.stack([_pose(translation_x=value) for value in translations])
    frame_count = len(translations)
    memory = _append_memory(
        latents=torch.arange(1, frame_count * height * width + 1, dtype=torch.float32).reshape(
            1, frame_count, height, width
        ),
        w2c=w2c,
        intrinsics=np.repeat(intrinsic[None], frame_count, axis=0),
        depths=np.ones((frame_count, height, width), dtype=np.float32),
    )
    query_w2c = np.repeat(_pose()[None], 4, axis=0)
    query_intrinsics = np.repeat(intrinsic[None], 4, axis=0)

    actual = memory.select_candidate_frame_ids(query_w2c, query_intrinsics)
    handler = _upstream_handler(upstream, memory)
    expected = handler.select_candidates(
        query_extrinsics=query_w2c,
        depths=memory.depths,
        w2c=memory.w2c,
        memory_K=memory.intrinsics,
        query_K=query_intrinsics,
        H_lat=height,
        W_lat=width,
        total_latents=frame_count,
        candidates_per_query_group=5,
        selection_mode="projection_iou",
        latent_merge_4frames=False,
        query_reference_frame=4,
        candidate_nms_mode="coverage",
        coverage_grid_downsample=4,
        coverage_pool_stride=2,
    )

    assert actual == tuple(tuple(group) for group in expected)
    assert actual == ((2, 4, 0, 6, 8),)


def test_far_z_fill_stop_matches_pinned_upstream_and_differs_from_near_z() -> None:
    upstream = _load_upstream()
    height = width = 8
    intrinsic = _intrinsic()
    # The shifted near frame covers seven of eight columns, below the 95%
    # stop ratio. The full far frame must therefore run and replace overlap.
    memory = _append_memory(
        latents=torch.stack(
            (
                torch.ones((height, width), dtype=torch.float32),
                torch.full((height, width), 9.0, dtype=torch.float32),
            )
        ).unsqueeze(0),
        w2c=np.stack((_pose(translation_x=2.0), _pose())),
        intrinsics=np.repeat(intrinsic[None], 2, axis=0),
        depths=np.stack(
            (
                np.full((height, width), 2.0, dtype=np.float32),
                np.full((height, width), 4.0, dtype=np.float32),
            )
        ),
    )
    query_w2c = np.repeat(_pose()[None], 4, axis=0)
    query_intrinsics = np.repeat(intrinsic[None], 4, axis=0)
    candidates = ((0, 1, -1, -1, -1),)

    actual_far, actual_valid = memory.fuse_candidates(
        query_w2c,
        query_intrinsics,
        candidates,
    )
    handler = _upstream_handler(upstream, memory)
    upstream_kwargs = dict(
        query_extrinsics=query_w2c,
        candidate_frame_ids=candidates,
        latents=memory.latents,
        w2c=memory.w2c,
        depths=memory.depths,
        memory_K=memory.intrinsics,
        query_K=query_intrinsics,
        fuse_mode="fill_stop_zbuffer",
        fill_ratio_threshold=0.95,
        interpolation_mode="nearest",
        latent_merge_4frames=False,
        query_reference_frame=4,
        fuse_block_size=1,
    )
    expected_far = handler.fuse_candidates(
        **upstream_kwargs,
        zbuffer_depth_preference="far",
    )
    expected_near = handler.fuse_candidates(
        **upstream_kwargs,
        zbuffer_depth_preference="near",
    )

    torch.testing.assert_close(actual_far, expected_far, rtol=0.0, atol=0.0)
    assert actual_valid.all()
    torch.testing.assert_close(expected_far, torch.full_like(expected_far, 9.0))
    torch.testing.assert_close(
        expected_near[..., :-1],
        torch.ones_like(expected_near[..., :-1]),
    )
    torch.testing.assert_close(
        expected_near[..., -1:],
        torch.full_like(expected_near[..., -1:], 9.0),
    )
    assert not torch.equal(expected_far, expected_near)
