# SPDX-License-Identifier: Apache-2.0
"""Narrow loaders for Matrix-Game 3.5's shared Wan components."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


def _verify_revision(reference_dir: Path) -> None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Cannot verify Matrix-Game 3.5 reference revision at {reference_dir}: {exc}"
        ) from exc
    if revision != PINNED_OFFICIAL_REVISION:
        raise RuntimeError(
            "Matrix-Game 3.5 reference revision mismatch: "
            f"expected {PINNED_OFFICIAL_REVISION}, got {revision}"
        )


def _load_source(reference_dir: str | Path, filename: str) -> ModuleType:
    reference_dir = Path(reference_dir).expanduser().resolve()
    _verify_revision(reference_dir)
    source_file = reference_dir / "diffsynth" / "models" / filename
    if not source_file.is_file():
        raise FileNotFoundError(f"Matrix-Game 3.5 source is missing: {source_file}")

    digest = hashlib.sha256(str(source_file).encode()).hexdigest()[:12]
    module_name = f"_matrixgame35_shared_{source_file.stem}_{digest}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import spec for {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_upstream_wan_vae(reference_dir: str | Path) -> ModuleType:
    """Load the real upstream WanVideoVAE38 module without umbrella imports."""

    return _load_source(reference_dir, "wan_video_vae.py")


def load_upstream_wan_text_encoder(reference_dir: str | Path) -> ModuleType:
    """Load the real upstream WanTextEncoder module without umbrella imports."""

    return _load_source(reference_dir, "wan_video_text_encoder.py")
