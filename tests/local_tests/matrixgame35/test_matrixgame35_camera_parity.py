# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.camera import (
    build_prope_viewmats,
    gather_latent_subframes,
    load_camera_trajectory,
    normalize_matrixgame35_intrinsics,
    required_camera_frames,
)
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


PARITY_SCOPE = "implementation_subcomponent"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"


def _load_upstream_dataset() -> ModuleType:
    revision = subprocess.run(
        ["git", "-C", str(_OFFICIAL_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_OFFICIAL_REVISION

    root_name = "_matrixgame35_camera_reference"
    packages = (
        (root_name, _OFFICIAL_DIR),
        (f"{root_name}.diffsynth", _OFFICIAL_DIR / "diffsynth"),
        (f"{root_name}.diffsynth.core", _OFFICIAL_DIR / "diffsynth" / "core"),
        (f"{root_name}.diffsynth.core.data", _OFFICIAL_DIR / "diffsynth" / "core" / "data"),
    )
    for name, path in packages:
        if name not in sys.modules:
            package = ModuleType(name)
            package.__package__ = name
            package.__path__ = [str(path)]
            sys.modules[name] = package

    module_name = f"{root_name}.diffsynth.core.data.unified_dataset"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = _OFFICIAL_DIR / "diffsynth" / "core" / "data" / "unified_dataset.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _poses(frame_count: int) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count, dtype=np.float32)
    return poses


@pytest.mark.parametrize(
    "intrinsics",
    (
        np.array([100.0, 120.0, 64.0, 32.0], dtype=np.float32),
        np.array([[100.0, 0.0, 64.0], [0.0, 120.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[100.0, 120.0, 64.0, 32.0], [101.0, 121.0, 65.0, 33.0]], dtype=np.float32),
    ),
)
def test_load_camera_formats_and_tail_padding(tmp_path, intrinsics) -> None:
    path = tmp_path / "camera.npz"
    np.savez(path, extrinsics_c2w=_poses(2), intrinsics=intrinsics)
    trajectory = load_camera_trajectory(path, frame_count=required_camera_frames(1))

    assert trajectory.c2w.shape == (85, 4, 4)
    assert trajectory.intrinsics.shape == (85, 3, 3)
    torch.testing.assert_close(trajectory.c2w[-1], trajectory.c2w[1])
    torch.testing.assert_close(trajectory.intrinsics[-1], trajectory.intrinsics[1])


def test_w2c_input_is_normalized_to_c2w(tmp_path) -> None:
    c2w = _poses(3)
    path = tmp_path / "camera.npz"
    np.savez(path, extrinsics=np.linalg.inv(c2w), intrinsics=np.array([100.0, 100.0, 50.0, 50.0]))
    trajectory = load_camera_trajectory(path, convention="w2c")
    torch.testing.assert_close(trajectory.c2w, torch.from_numpy(c2w))


def test_gather_one_block_uses_21_groups_of_four_after_anchor(tmp_path) -> None:
    path = tmp_path / "camera.npz"
    np.savez(path, extrinsics_c2w=_poses(85), intrinsics=np.array([100.0, 100.0, 50.0, 50.0]))
    trajectory = load_camera_trajectory(path)
    c2w, intrinsics = gather_latent_subframes(trajectory, first_rgb_frame=1, latent_count=21)

    assert c2w.shape == (1, 21, 4, 4, 4)
    assert intrinsics.shape == (1, 21, 4, 3, 3)
    torch.testing.assert_close(c2w[0, 0, :, 0, 3], torch.arange(1, 5, dtype=torch.float32))
    torch.testing.assert_close(c2w[0, -1, :, 0, 3], torch.arange(81, 85, dtype=torch.float32))


@pytest.mark.parametrize("mode", ("per_frame", "first_frame", "episode_mean"))
def test_intrinsics_normalization_matches_pinned_dataset(mode: str) -> None:
    if not _OFFICIAL_DIR.is_dir():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {_OFFICIAL_DIR}")
    upstream_module = _load_upstream_dataset()
    upstream_dataset = object.__new__(upstream_module._MosaicDatasetBase)
    upstream_dataset.mosaic_intrinsics_mode = mode
    intrinsics = np.array(
        (
            ((900.0, 12.0, 450.0), (0.0, 700.0, 350.0), (0.0, 0.0, 1.0)),
            ((800.0, 8.0, 400.0), (0.0, 600.0, 300.0), (0.0, 0.0, 1.0)),
            ((1000.0, 16.0, 500.0), (0.0, 800.0, 400.0), (0.0, 0.0, 1.0)),
        ),
        dtype=np.float64,
    )

    expected = upstream_dataset.normalize_and_scale_intrinsics(
        intrinsics,
        H_img=704,
        W_img=1280,
        temporal_mean=upstream_dataset._intrinsics_temporal_mean_enabled(),
    )
    actual = normalize_matrixgame35_intrinsics(
        intrinsics,
        image_height=704,
        image_width=1280,
        mode=mode,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("translation_scale", (50.0, "logd4"))
def test_prope_matrix_composition_matches_upstream_formula(translation_scale) -> None:
    c2w = torch.eye(4, dtype=torch.float32).reshape(1, 1, 1, 4, 4).repeat(1, 2, 4, 1, 1)
    c2w[..., 0, 3] = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 1, 4).repeat(1, 2, 1)
    intrinsics = torch.zeros(1, 2, 4, 3, 3, dtype=torch.float32)
    intrinsics[..., 0, 0] = 640.0
    intrinsics[..., 1, 1] = 352.0
    intrinsics[..., 0, 2] = 640.0
    intrinsics[..., 1, 2] = 352.0
    intrinsics[..., 2, 2] = 1.0

    projection, transpose, inverse = build_prope_viewmats(
        c2w,
        intrinsics,
        image_height=704,
        image_width=1280,
        translation_scale=translation_scale,
        dtype=torch.float32,
    )

    w2c = torch.linalg.inv(c2w.double()).float()
    translation = w2c[..., :3, 3]
    if translation_scale == "logd4":
        norm = translation.norm(dim=-1, keepdim=True)
        translation = translation * torch.log1p(norm) / norm.clamp_min(1e-8) / 4.0
    else:
        translation = translation / translation_scale
    w2c[..., :3, 3] = translation
    normalized = torch.zeros_like(intrinsics)
    normalized[..., 0, 0] = 0.5
    normalized[..., 1, 1] = 0.5
    normalized[..., 2, 2] = 1.0
    lifted = torch.zeros(*normalized.shape[:-2], 4, 4)
    lifted[..., :3, :3] = normalized
    lifted[..., 3, 3] = 1.0
    expected_projection = lifted @ w2c

    torch.testing.assert_close(projection, expected_projection)
    torch.testing.assert_close(transpose, expected_projection.transpose(-1, -2))
    torch.testing.assert_close(projection @ inverse, torch.eye(4).expand_as(projection), atol=1e-5, rtol=1e-5)


def test_rejects_invalid_camera_contract(tmp_path) -> None:
    path = tmp_path / "camera.npz"
    np.savez(path, extrinsics_c2w=np.ones((2, 3, 4), dtype=np.float32), intrinsics=np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="extrinsics must have shape"):
        load_camera_trajectory(path)
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        required_camera_frames(0)
