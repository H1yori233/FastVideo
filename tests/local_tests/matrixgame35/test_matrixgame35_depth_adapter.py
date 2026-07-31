# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the optional Matrix-Game 3.5 DA3 adapter."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from fastvideo.pipelines.basic.matrixgame35._depth_anything3 import (
    DA3_PROCESS_RES,
    MatrixGame35DepthAnything3Adapter,
)
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5"))


class _FakeDepthEstimator:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth
        self.calls: list[tuple[list[Image.Image], dict[str, object]]] = []
        self.devices: list[torch.device] = []
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.devices.append(torch.device(device))
        return self

    def inference(self, frames, **kwargs):
        assert not torch.is_autocast_enabled("cpu")
        self.calls.append((frames, kwargs))
        return SimpleNamespace(depth=self.depth)


@pytest.mark.parametrize("config_name", ("infer_first_person.yaml", "infer_third_person.yaml"))
def test_base_da3_resolution_matches_pinned_release(config_name: str) -> None:
    config_path = OFFICIAL_REF_DIR / "configs" / config_name
    if not config_path.is_file():
        pytest.skip(
            "Pinned Matrix-Game 3.5 configs are absent; set "
            f"MATRIXGAME35_OFFICIAL_REF_DIR to commit {PINNED_OFFICIAL_REVISION}."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["vggt_omega_da3_process_res"] == DA3_PROCESS_RES == 504


def test_publication_batch_is_one_depth_only_da3_call() -> None:
    raw_depth = np.arange(84 * 2 * 4, dtype=np.float64).reshape(84, 2, 4)[:, :, ::-1]
    estimator = _FakeDepthEstimator(raw_depth)
    loaded_refs: list[str] = []

    def load_estimator(model_ref: str) -> _FakeDepthEstimator:
        loaded_refs.append(model_ref)
        return estimator

    adapter = MatrixGame35DepthAnything3Adapter(
        "local/da3",
        device="cpu",
        estimator_loader=load_estimator,
    )
    assert loaded_refs == []

    frames = [np.full((3, 5, 3), index % 256, dtype=np.uint8) for index in range(84)]
    depths = adapter.estimate_depth(frames)

    assert loaded_refs == ["local/da3"]
    assert estimator.eval_calls == 1
    assert estimator.devices == [torch.device("cpu")]
    assert len(estimator.calls) == 1
    called_frames, called_kwargs = estimator.calls[0]
    assert len(called_frames) == 84
    assert all(isinstance(frame, Image.Image) and frame.mode == "RGB" for frame in called_frames)
    assert DA3_PROCESS_RES == 504
    assert called_kwargs == {"use_ray_pose": False, "process_res": DA3_PROCESS_RES}
    assert depths.shape == (84, 2, 4)
    assert depths.dtype == np.float32
    assert depths.flags.c_contiguous
    np.testing.assert_array_equal(depths, raw_depth.astype(np.float32))


def test_anchor_normalizes_pil_and_float_numpy_to_uint8_rgb() -> None:
    estimator = _FakeDepthEstimator(np.ones((2, 4, 6), dtype=np.float32))
    adapter = MatrixGame35DepthAnything3Adapter(device="cpu", estimator=estimator)
    grayscale = Image.fromarray(np.zeros((4, 6), dtype=np.uint8), mode="L")
    normalized = np.full((4, 6, 3), 0.5, dtype=np.float32)

    adapter.estimate_depth([grayscale, normalized])

    called_frames, _ = estimator.calls[0]
    assert called_frames[0].mode == "RGB"
    assert np.asarray(called_frames[1]).dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(called_frames[1]), np.full((4, 6, 3), 127, dtype=np.uint8))


def test_cpu_offload_parks_estimator_after_each_call() -> None:
    estimator = _FakeDepthEstimator(np.ones((1, 2, 3), dtype=np.float32))
    adapter = MatrixGame35DepthAnything3Adapter(
        device="cpu",
        estimator=estimator,
        cpu_offload=True,
    )

    adapter.estimate_depth([np.zeros((2, 3, 3), dtype=np.uint8)])

    assert estimator.devices == [torch.device("cpu"), torch.device("cpu")]


@pytest.mark.parametrize(
    ("frames", "error", "match"),
    [
        ([], ValueError, "at least one"),
        ([np.zeros((2, 3), dtype=np.uint8)], ValueError, "height, width, 3"),
        ([object()], TypeError, "NumPy arrays or PIL images"),
    ],
)
def test_invalid_frames_fail_before_loading(frames, error, match) -> None:
    loader_calls = 0

    def loader(_model_ref: str):
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("loader should not run")

    adapter = MatrixGame35DepthAnything3Adapter(device="cpu", estimator_loader=loader)
    with pytest.raises(error, match=match):
        adapter.estimate_depth(frames)
    assert loader_calls == 0


def test_depth_shape_must_match_publication_batch() -> None:
    estimator = _FakeDepthEstimator(np.ones((1, 2, 3), dtype=np.float32))
    adapter = MatrixGame35DepthAnything3Adapter(device="cpu", estimator=estimator)
    frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(84)]

    with pytest.raises(ValueError, match=r"\[84, height, width\]"):
        adapter.estimate_depth(frames)


RUN_REAL_DA3 = os.environ.get("FASTVIDEO_MATRIXGAME35_RUN_DA3_CUDA") == "1"


@pytest.mark.skipif(
    not RUN_REAL_DA3,
    reason="set FASTVIDEO_MATRIXGAME35_RUN_DA3_CUDA=1 with DA3 model and image assets",
)
def test_real_depth_anything3_cuda_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.fail("FASTVIDEO_MATRIXGAME35_RUN_DA3_CUDA=1 requires a CUDA device")
    model_ref = os.environ.get("FASTVIDEO_MATRIXGAME35_DA3_MODEL_PATH")
    image_path = os.environ.get("FASTVIDEO_MATRIXGAME35_DA3_IMAGE")
    if not model_ref or not Path(model_ref).exists():
        pytest.fail("FASTVIDEO_MATRIXGAME35_DA3_MODEL_PATH must point to the local DA3 model")
    if not image_path or not Path(image_path).is_file():
        pytest.fail("FASTVIDEO_MATRIXGAME35_DA3_IMAGE must point to an RGB test image")

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        expected_shape = (rgb.height, rgb.width)
        adapter = MatrixGame35DepthAnything3Adapter(
            model_ref,
            device=torch.device("cuda", torch.cuda.current_device()),
            cpu_offload=True,
        )
        depths = adapter.estimate_depth([rgb])

    assert depths.shape == (1, *expected_shape)
    assert depths.dtype == np.float32
    assert depths.flags.c_contiguous
    assert np.isfinite(depths).all()
    assert (depths > 0).any()
