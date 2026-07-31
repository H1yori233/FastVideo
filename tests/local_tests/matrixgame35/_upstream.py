# SPDX-License-Identifier: Apache-2.0
"""Narrow importer for the pinned Matrix-Game 3.5 transformer sources.

The upstream ``diffsynth`` package eagerly imports optional download and media
dependencies from its package initializers. Transformer parity needs none of
them, so this helper creates isolated namespace packages and executes the real
transformer dependency graph without executing those umbrella initializers.
No numerical module or kernel is stubbed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType


PINNED_OFFICIAL_REVISION = "fa6d2b628ac9b0f1657dc24689536d74bfeb51da"


@dataclass(frozen=True)
class UpstreamTransformerModules:
    """Real upstream modules loaded under an isolated namespace."""

    wan_video_dit: ModuleType
    prope_attention: ModuleType


def _git_revision(reference_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Cannot verify Matrix-Game 3.5 reference revision at {reference_dir}: {exc}"
        ) from exc


def _install_namespace(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_paths = tuple(Path(item).resolve() for item in getattr(existing, "__path__", ()))
        if path.resolve() not in existing_paths:
            raise RuntimeError(
                f"Upstream import namespace {name!r} already points to {existing_paths}, "
                f"not {path.resolve()}"
            )
        return

    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__file__ = str(path)
    sys.modules[name] = package


def load_upstream_transformer(
    reference_dir: str | Path,
    *,
    verify_revision: bool = True,
) -> UpstreamTransformerModules:
    """Load the real upstream Wan DiT and PRoPE modules without umbrella imports."""

    reference_dir = Path(reference_dir).expanduser().resolve()
    source_file = reference_dir / "diffsynth" / "models" / "wan_video_dit.py"
    if not source_file.is_file():
        raise FileNotFoundError(
            f"Matrix-Game 3.5 transformer source is missing: {source_file}"
        )

    if verify_revision:
        revision = _git_revision(reference_dir)
        if revision != PINNED_OFFICIAL_REVISION:
            raise RuntimeError(
                "Matrix-Game 3.5 reference revision mismatch: "
                f"expected {PINNED_OFFICIAL_REVISION}, got {revision}"
            )

    path_digest = hashlib.sha256(str(reference_dir).encode()).hexdigest()[:12]
    root_name = f"_matrixgame35_upstream_{path_digest}"
    diffsynth_name = f"{root_name}.diffsynth"
    models_name = f"{diffsynth_name}.models"
    core_name = f"{diffsynth_name}.core"

    _install_namespace(root_name, reference_dir)
    _install_namespace(diffsynth_name, reference_dir / "diffsynth")
    _install_namespace(models_name, reference_dir / "diffsynth" / "models")
    _install_namespace(core_name, reference_dir / "diffsynth" / "core")

    module_name = f"{models_name}.wan_video_dit"
    try:
        wan_video_dit = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preserve the direct import failure.
        raise RuntimeError(
            f"Narrow Matrix-Game 3.5 transformer import failed from {source_file}: {exc}"
        ) from exc

    loaded_file = Path(wan_video_dit.__file__).resolve()
    if loaded_file != source_file.resolve():
        raise RuntimeError(
            f"Loaded unexpected Matrix-Game transformer source {loaded_file}; "
            f"expected {source_file.resolve()}"
        )

    prope_attention = importlib.import_module(f"{models_name}.prope_attention")
    return UpstreamTransformerModules(
        wan_video_dit=wan_video_dit,
        prope_attention=prope_attention,
    )
