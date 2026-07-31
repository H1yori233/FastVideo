# SPDX-License-Identifier: Apache-2.0
"""Narrow loader for the pinned Matrix-Game 3.5 Patch Memory source."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np

from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


def _git_revision(reference_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _nearest_resize(array: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    if interpolation != 0:
        raise RuntimeError("The Patch Memory parity loader only supports INTER_NEAREST.")
    width, height = size
    rows = np.floor(np.arange(height, dtype=np.float64) * array.shape[0] / height).astype(np.int64)
    columns = np.floor(np.arange(width, dtype=np.float64) * array.shape[1] / width).astype(np.int64)
    return np.ascontiguousarray(array[rows[:, None], columns[None, :]])


def load_upstream_patch_memory(
    reference_dir: str | Path,
    *,
    verify_revision: bool = True,
) -> ModuleType:
    """Execute the real frustum handler while isolating its optional cv2 import."""
    reference_dir = Path(reference_dir).expanduser().resolve()
    source_file = reference_dir / "frustum" / "frustum_handler.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"Matrix-Game 3.5 Patch Memory source is missing: {source_file}")
    if verify_revision:
        revision = _git_revision(reference_dir)
        if revision != PINNED_OFFICIAL_REVISION:
            raise RuntimeError(
                "Matrix-Game 3.5 reference revision mismatch: "
                f"expected {PINNED_OFFICIAL_REVISION}, got {revision}"
            )

    module_name = f"_matrixgame35_patch_memory_{hashlib.sha256(str(reference_dir).encode()).hexdigest()[:12]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    inserted_cv2 = "cv2" not in sys.modules
    if inserted_cv2:
        cv2 = ModuleType("cv2")
        cv2.INTER_NEAREST = 0
        cv2.resize = _nearest_resize
        sys.modules["cv2"] = cv2
    try:
        spec = importlib.util.spec_from_file_location(module_name, source_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import pinned Patch Memory source: {source_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if inserted_cv2:
            sys.modules.pop("cv2", None)

    if Path(module.__file__).resolve() != source_file:
        raise RuntimeError(f"Loaded unexpected Patch Memory source {module.__file__}; expected {source_file}.")
    return module
