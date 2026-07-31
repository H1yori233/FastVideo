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
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any


PINNED_OFFICIAL_REVISION = "fa6d2b628ac9b0f1657dc24689536d74bfeb51da"


@dataclass(frozen=True)
class UpstreamTransformerModules:
    """Real upstream modules loaded under an isolated namespace."""

    wan_video_dit: ModuleType
    prope_attention: ModuleType


class _ImportOnlyType:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


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


def _install_namespace(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_paths = tuple(Path(item).resolve() for item in getattr(existing, "__path__", ()))
        if path.resolve() not in existing_paths:
            raise RuntimeError(
                f"Upstream import namespace {name!r} already points to {existing_paths}, "
                f"not {path.resolve()}"
            )
        return existing

    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__file__ = str(path)
    sys.modules[name] = package
    return package


def _install_module(name: str, **attributes: Any) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = module
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


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


def load_upstream_pipeline(
    reference_dir: str | Path,
    *,
    verify_revision: bool = True,
) -> ModuleType:
    """Load the real upstream Wan pipeline function with import-only side modules."""
    reference_dir = Path(reference_dir).expanduser().resolve()
    transformer_modules = load_upstream_transformer(
        reference_dir,
        verify_revision=verify_revision,
    )
    models_name = transformer_modules.wan_video_dit.__name__.rsplit(".", 1)[0]
    diffsynth_name = models_name.rsplit(".", 1)[0]
    source_file = reference_dir / "diffsynth" / "pipelines" / "wan_video.py"

    core = sys.modules[f"{diffsynth_name}.core"]
    core.ModelConfig = _ImportOnlyType

    def direct_forward(module, _checkpointing, _offload, *args):
        return module(*args)

    core.gradient_checkpoint_forward = direct_forward
    _install_namespace(
        f"{diffsynth_name}.core.device",
        reference_dir / "diffsynth" / "core" / "device",
    )
    _install_module(
        f"{diffsynth_name}.core.device.npu_compatible_device",
        get_device_type=lambda: "cpu",
    )
    _install_module(f"{diffsynth_name}.diffusion", FlowMatchScheduler=_ImportOnlyType)
    _install_module(
        f"{diffsynth_name}.diffusion.base_pipeline",
        BasePipeline=_ImportOnlyType,
        PipelineUnit=_ImportOnlyType,
    )

    import_only_models = {
        "wan_video_dit_s2v": {"rope_precompute": lambda *_args, **_kwargs: None},
        "wan_video_text_encoder": {
            "WanTextEncoder": _ImportOnlyType,
            "HuggingfaceTokenizer": _ImportOnlyType,
        },
        "wan_video_vae": {"WanVideoVAE": _ImportOnlyType},
        "wan_video_image_encoder": {"WanImageEncoder": _ImportOnlyType},
        "wan_video_vace": {"VaceWanModel": _ImportOnlyType},
        "wan_video_motion_controller": {"WanMotionControllerModel": _ImportOnlyType},
        "wan_video_animate_adapter": {"WanAnimateAdapter": _ImportOnlyType},
        "wan_video_mot": {"MotWanModel": _ImportOnlyType},
        "wav2vec": {"WanS2VAudioEncoder": _ImportOnlyType},
        "longcat_video_dit": {"LongCatVideoTransformer3DModel": _ImportOnlyType},
    }
    for module_name, attributes in import_only_models.items():
        _install_module(f"{models_name}.{module_name}", **attributes)

    _install_namespace(
        f"{diffsynth_name}.pipelines",
        reference_dir / "diffsynth" / "pipelines",
    )
    module_name = f"{diffsynth_name}.pipelines.wan_video"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned upstream pipeline source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source_file:
        raise RuntimeError(
            f"Loaded unexpected Matrix-Game pipeline source {module.__file__}; expected {source_file}."
        )
    return module
