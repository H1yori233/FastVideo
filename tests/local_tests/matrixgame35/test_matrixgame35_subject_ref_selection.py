# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from PIL import Image

from fastvideo.pipelines.basic.matrixgame35.subject_refs import (
    _sample_reference_indices,
    _select_reference_indices,
    select_subject_references,
)
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"
_SELECTION_SEED = 3750906216
_PAIRWISE = np.asarray(
    [
        [1.0, 0.9, 0.2, 0.8],
        [0.9, 1.0, 0.3, 0.1],
        [0.2, 0.3, 1.0, 0.7],
        [0.8, 0.1, 0.7, 1.0],
    ],
    dtype=np.float32,
)


def _write_rgb(path: Path, value: int) -> None:
    Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _write_export(path: Path, rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        image_path = path / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _write_rgb(image_path, index)
        if row.get("frame_idx") is not None:
            row["mask_path"] = f"mask_{index}.png"
            Image.fromarray(np.full((2, 2), 255, dtype=np.uint8), mode="L").save(path / row["mask_path"])
    (path / "candidates.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("candidate_count", "expected"),
    (
        (0, []),
        (1, [0]),
        (2, [1, 0]),
        (3, [1, 0, 2]),
        (4, [1, 2, 0, 3]),
        (5, [1, 0, 4, 3]),
    ),
)
def test_no_pairwise_selection_matches_released_fixed_vectors(candidate_count: int, expected: list[int]) -> None:
    assert _select_reference_indices(candidate_count, None, max_refs=4) == expected


def test_pairwise_selection_and_final_shuffle_match_released_fixed_vector() -> None:
    assert _select_reference_indices(4, _PAIRWISE, max_refs=4) == [2, 0, 1, 3]


@pytest.mark.parametrize(
    ("pairwise", "expected"),
    (
        (np.zeros((3, 3), dtype=np.float32), [1, 2, 0, 3]),
        (
            np.asarray(
                [
                    [1.0, np.nan, 0.2, 0.8],
                    [np.inf, 1.0, 0.3, 0.1],
                    [-np.inf, 0.3, 1.0, 0.7],
                    [0.8, 0.1, 0.7, 1.0],
                ],
                dtype=np.float32,
            ),
            [1, 2, 0, 3],
        ),
    ),
)
def test_stale_and_nonfinite_pairwise_inputs_match_released_fallbacks(
    pairwise: np.ndarray,
    expected: list[int],
) -> None:
    assert _select_reference_indices(4, pairwise, max_refs=4) == expected


def test_export_filters_rows_without_frame_index_before_selection(tmp_path: Path) -> None:
    rows = [
        {"image_path": "ignored.png"},
        *({"frame_idx": index, "image_path": f"ref_{index}.png"} for index in range(4)),
    ]
    _write_export(tmp_path, rows)

    selected = select_subject_references(tmp_path, max_refs=4)

    assert [reference.image_path.name for reference in selected] == ["ref_1.png", "ref_2.png", "ref_0.png", "ref_3.png"]


def test_plain_directory_matches_public_wrapper_selection(tmp_path: Path) -> None:
    for index, name in enumerate(("c.png", "a.png", "b.png")):
        _write_rgb(tmp_path / name, index)

    selected = select_subject_references(tmp_path, max_refs=3)

    assert [reference.image_path.name for reference in selected] == ["b.png", "a.png", "c.png"]


def _load_pinned_upstream_sampler():
    source = _OFFICIAL_DIR / "diffsynth" / "core" / "data" / "subject_ref_memory_dataset.py"
    if not source.is_file():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {source}")
    revision = subprocess.run(
        ["git", "-C", str(_OFFICIAL_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_OFFICIAL_REVISION:
        pytest.skip(f"Matrix-Game 3.5 source is not pinned to {PINNED_OFFICIAL_REVISION}: {revision}")

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sample_reference_indices"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_sample_reference_indices"]


def test_pairwise_sampler_matches_pinned_upstream_source() -> None:
    upstream_sampler = _load_pinned_upstream_sampler()
    kwargs = {
        "num_refs": 4,
        "dissimilar_top_k": 8,
        "max_similarity": 0.94,
    }

    expected = upstream_sampler(4, _PAIRWISE, rng=np.random.default_rng(_SELECTION_SEED), **kwargs)
    actual = _sample_reference_indices(4, _PAIRWISE, rng=np.random.default_rng(_SELECTION_SEED), **kwargs)

    assert actual == expected == [1, 2, 3, 0]
