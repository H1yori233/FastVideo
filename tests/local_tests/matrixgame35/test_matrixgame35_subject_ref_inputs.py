# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import torch
from PIL import Image

from fastvideo.pipelines.basic.matrixgame35.subject_refs import (
    IMAGENET_MEAN_RGB,
    build_subject_reference_canvas,
    discover_subject_references,
    load_subject_reference_canvases,
)


def _write_rgb(path, rgb):
    Image.fromarray(rgb, mode="RGB").save(path)


def test_plain_directory_discovery_ignores_mask_images(tmp_path):
    rgb = np.full((4, 5, 3), 17, dtype=np.uint8)
    _write_rgb(tmp_path / "b.png", rgb)
    _write_rgb(tmp_path / "a.png", rgb)
    Image.fromarray(np.full((4, 5), 255, dtype=np.uint8), mode="L").save(tmp_path / "a_mask.png")

    refs = discover_subject_references(tmp_path)

    assert [ref.image_path.name for ref in refs] == ["a.png", "b.png"]
    assert refs[0].mask_path == tmp_path / "a_mask.png"
    assert refs[1].mask_path is None


def test_official_candidates_paths_are_resolved_relative_to_export(tmp_path):
    image_dir = tmp_path / "ref_images"
    image_dir.mkdir()
    _write_rgb(image_dir / "ref.png", np.zeros((2, 2, 3), dtype=np.uint8))
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8), mode="L").save(image_dir / "mask.png")
    row = {"frame_idx": 0, "image_path": "ref_images/ref.png", "mask_path": "ref_images/mask.png"}
    (tmp_path / "candidates.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    refs = discover_subject_references(tmp_path)

    assert refs[0].image_path == image_dir / "ref.png"
    assert refs[0].mask_path == image_dir / "mask.png"


def test_canvas_matches_released_bottom_right_slot_contract(tmp_path):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    _write_rgb(tmp_path / "ref.png", rgb)
    reference = discover_subject_references(tmp_path)[0]

    canvas = build_subject_reference_canvas(reference, height=8, width=12, slot_ratio=0.5)

    expected_background = torch.from_numpy(IMAGENET_MEAN_RGB).mul(2.0 / 255.0).sub(1.0)
    torch.testing.assert_close(canvas[:, 0, 0], expected_background)
    torch.testing.assert_close(canvas[:, -1, -1], torch.tensor([1.0, -1.0, -1.0]))
    assert canvas.shape == (3, 8, 12)


def test_loader_caps_reference_count(tmp_path):
    for index in range(3):
        _write_rgb(tmp_path / f"ref_{index}.png", np.full((2, 2, 3), index, dtype=np.uint8))

    canvases = load_subject_reference_canvases(tmp_path, height=8, width=8, max_refs=2)

    assert canvases.shape == (2, 3, 8, 8)


def test_invalid_or_empty_sources_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="No subject reference"):
        discover_subject_references(tmp_path)
    with pytest.raises(ValueError, match="slot_ratio"):
        build_subject_reference_canvas(
            type("Ref", (), {"image_path": tmp_path / "missing.png", "mask_path": None})(),
            height=8,
            width=8,
            slot_ratio=0.0,
        )
