# SPDX-License-Identifier: Apache-2.0
"""CPU differential tests for Matrix-Game 3.5 Base Patch Memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.patch_memory import (
    MatrixGame35BasePatchMemory,
    _nearest_resize_depth,
)
from tests.local_tests.matrixgame35._upstream_patch_memory import (
    load_upstream_patch_memory,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = REPO_ROOT / "Matrix-Game-3.5"
PARITY_SCOPE = "implementation_subcomponent"


def _pose(*, center_x: float = 0.0, yaw_degrees: float = 0.0, z_translation: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_degrees)
    rotation = np.array(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        ),
        dtype=np.float32,
    )
    center = np.array((center_x, 0.0, 0.0), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ center
    pose[2, 3] += np.float32(z_translation)
    return pose


def _intrinsic() -> np.ndarray:
    return np.array(((16.0, 0.0, 0.0), (0.0, 16.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)


def _make_memory(
    w2c: np.ndarray,
    *,
    height: int = 3,
    width: int = 4,
    depths: np.ndarray | None = None,
) -> MatrixGame35BasePatchMemory:
    frame_count = int(w2c.shape[0])
    latents = torch.arange(1, 2 * frame_count * height * width + 1, dtype=torch.float32).reshape(
        2, frame_count, height, width
    )
    if depths is None:
        depths = np.full((frame_count, height, width), 2.0, dtype=np.float32)
    memory = MatrixGame35BasePatchMemory()
    memory.append(
        latents=latents,
        w2c=w2c,
        intrinsics=np.repeat(_intrinsic()[None], frame_count, axis=0),
        depths=depths,
    )
    return memory


def _upstream_handler(module, memory: MatrixGame35BasePatchMemory):
    handler = module.FrustumHandler(
        _intrinsic(),
        image_size=(memory.depths.shape[-2], memory.depths.shape[-1]),
        latent_stride=16,
        use_gpu=False,
    )
    handler._Kpix = memory.intrinsics[-1].copy()
    handler._Kpixs = [item.copy() for item in memory.intrinsics]
    return handler


def test_append_is_frame_aligned_cpu_fp32_and_uses_depth_adapter() -> None:
    class DepthAdapter:
        def __init__(self) -> None:
            self.call_count = 0

        def estimate_depth(self, frames) -> np.ndarray:
            self.call_count += 1
            return np.stack([np.full((4, 6), index + 1, dtype=np.float64) for index in range(len(frames))])

    adapter = DepthAdapter()
    memory = MatrixGame35BasePatchMemory()
    source = torch.ones(1, 2, 2, 3, dtype=torch.float16)
    memory.append(
        latents=source,
        w2c=np.stack((_pose(), _pose(center_x=0.2))).astype(np.float64),
        intrinsics=np.repeat(_intrinsic()[None], 2, axis=0).astype(np.float64),
        frames=[np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)],
        depth_adapter=adapter,
    )
    source.zero_()
    memory.append(
        latents=torch.full((1, 1, 2, 3), 3.0),
        w2c=_pose(center_x=0.4)[None],
        intrinsics=_intrinsic()[None],
        depths=np.ones((1, 4, 6), dtype=np.float64),
    )

    assert adapter.call_count == 1
    assert memory.num_frames == 3
    assert memory.latents.device.type == "cpu"
    assert memory.latents.dtype == torch.float32
    assert memory.w2c.dtype == np.float32
    assert memory.intrinsics.dtype == np.float32
    assert memory.depths.dtype == np.float32
    torch.testing.assert_close(memory.latents[:, :2], torch.ones(1, 2, 2, 3))


def test_nearest_depth_downsample_uses_source_grid_samples() -> None:
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)
    actual = _nearest_resize_depth(depth, 2, 3)
    np.testing.assert_array_equal(actual, depth[np.ix_([0, 2], [0, 2, 4])])


def test_anchor_alignment_and_pose_nearest_nms_match_pinned_upstream() -> None:
    upstream = load_upstream_patch_memory(OFFICIAL_REF_DIR)
    memory_w2c = np.stack(
        [_pose(center_x=index * 0.04, yaw_degrees=(index % 4) * 3.0) for index in range(14)]
    )
    memory = _make_memory(memory_w2c)
    anchor = _pose(center_x=-2.0, yaw_degrees=6.0)
    raw_query = np.stack(
        [_pose(center_x=-2.0 + index * 0.015, yaw_degrees=5.0 + (index % 7)) for index in range(84)]
    )

    actual_aligned = memory.align_query_trajectory(anchor, raw_query)
    handler = _upstream_handler(upstream, memory)
    expected_aligned = handler.align_w2c_trajectory(
        np.concatenate((anchor[None], raw_query), axis=0),
        memory.w2c[-1],
    )[1:]
    np.testing.assert_allclose(actual_aligned, expected_aligned, rtol=1e-5, atol=1e-6)

    actual_ids = memory.select_candidate_frame_ids(actual_aligned)
    expected_ids = handler.select_candidates(
        query_extrinsics=expected_aligned,
        depths=memory.depths,
        w2c=memory.w2c,
        H_lat=memory.latents.shape[-2],
        W_lat=memory.latents.shape[-1],
        total_latents=memory.num_frames,
        candidates_per_query_group=5,
        selection_mode="pose_nearest",
        latent_merge_4frames=False,
        query_reference_frame=4,
        candidate_nms_mode="pose",
        candidate_nms_pose_distance_threshold=0.1,
        candidate_nms_pool_multiplier=2.0,
    )
    assert actual_ids == tuple(tuple(group) for group in expected_ids)
    assert len(actual_ids) == 21


def test_pose_nms_exhaustion_falls_back_to_nearest_ranked_candidates() -> None:
    memory = _make_memory(np.stack([_pose(center_x=index * 0.01) for index in range(6)]))
    query = np.repeat(_pose(center_x=0.0)[None], 84, axis=0)
    selected = memory.select_candidate_frame_ids(query)

    assert selected[0] == (0, 1, 2, 3, 4)
    assert all(group == selected[0] for group in selected)


def test_forward_warp_per_candidate_and_fill_stop_zbuffer_match_upstream() -> None:
    upstream = load_upstream_patch_memory(OFFICIAL_REF_DIR)
    memory_w2c = np.stack((_pose(), _pose()))
    depths = np.empty((2, 3, 4), dtype=np.float32)
    depths[0] = np.array((4.0, 5.0, 6.0, 7.0), dtype=np.float32)[None]
    depths[1] = np.array((1.0, 2.0, 3.0, 4.0), dtype=np.float32)[None]
    memory = _make_memory(memory_w2c, depths=depths)
    aligned_query = np.repeat(_pose(z_translation=4.0)[None], 84, axis=0)
    aligned_query[-1] = _pose(center_x=100.0, z_translation=4.0)
    candidates = tuple((0, 1, -1, -1, -1) for _ in range(21))
    query_intrinsics = np.repeat(_intrinsic()[None], 84, axis=0)

    actual, actual_valid = memory.fuse_candidates(
        aligned_query,
        candidates,
        query_intrinsics=query_intrinsics,
    )
    handler = _upstream_handler(upstream, memory)
    expected = handler.fuse_candidates(
        query_extrinsics=aligned_query,
        candidate_frame_ids=candidates,
        latents=memory.latents,
        w2c=memory.w2c,
        depths=memory.depths,
        memory_K=memory.intrinsics,
        query_K=query_intrinsics,
        fuse_mode="fill_stop_zbuffer",
        fill_ratio_threshold=0.95,
        zbuffer_depth_preference="near",
        interpolation_mode="nearest",
        latent_merge_4frames=False,
        query_reference_frame=4,
        fuse_block_size=1,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_valid, expected.abs().sum(dim=0) > 0)
    assert torch.count_nonzero(actual[:, -1]) == 0


def test_fill_ratio_stops_after_a_full_first_candidate() -> None:
    memory = _make_memory(np.stack((_pose(), _pose())))
    memory.latents[:, 0].fill_(1.0)
    memory.latents[:, 1].fill_(9.0)
    memory._depths[0].fill(4.0)
    memory._depths[1].fill(1.0)
    query = np.repeat(_pose()[None], 84, axis=0)
    candidates = tuple((0, 1, -1, -1, -1) for _ in range(21))

    fused, valid = memory.fuse_candidates(query, candidates)

    torch.testing.assert_close(fused, torch.ones_like(fused))
    assert valid.all()


def test_base_query_rejects_non_84_frame_blocks() -> None:
    memory = _make_memory(_pose()[None])
    with pytest.raises(ValueError, match="exactly 84"):
        memory.align_query_trajectory(_pose(), np.repeat(_pose()[None], 83, axis=0))
