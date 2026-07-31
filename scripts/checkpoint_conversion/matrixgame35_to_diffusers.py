#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the three released Matrix-Game 3.5 DiTs to FastVideo format.

The public checkpoints are raw, unsharded safetensors files. Their primary key
namespace is unprefixed; fully prefixed ``pipe.dit.`` and ``dit.`` checkpoints
are accepted only as all-or-nothing compatibility layouts. The architecture
config owns the official-to-FastVideo parameter mapping.

Examples:
    python scripts/checkpoint_conversion/matrixgame35_to_diffusers.py \
        --src official_weights/matrixgame35 \
        --dst converted_weights/matrixgame35 \
        --verify-sha256

    python scripts/checkpoint_conversion/matrixgame35_to_diffusers.py \
        --src official_weights/matrixgame35 \
        --dst converted_weights/matrixgame35 \
        --component-source official_weights/Wan2.2-TI2V-5B-Diffusers \
        --component-revision b8fff7315c768468a5333511427288870b2e9635 \
        --verify-sha256

    python scripts/checkpoint_conversion/matrixgame35_to_diffusers.py \
        --src official_weights/matrixgame35/base/first-person.safetensors \
        --dst converted_weights/matrixgame35 \
        --variant base_first_person
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastvideo.configs.models.dits.matrixgame35 import MatrixGame35WanVideoConfig
from fastvideo.models.loader.utils import get_param_names_mapping


OFFICIAL_COMPAT_PREFIXES = ("pipe.dit.", "dit.")
SUBJECT_REF_KEYS = (
    "subject_ref_index_embedding",
    "subject_ref_type_embedding",
    "subject_ref_local_h_embedding",
    "subject_ref_local_w_embedding",
)
SKIP_PATTERNS: tuple[str, ...] = ()

WAN22_COMPONENT_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN22_COMPONENT_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
WAN22_DIFFUSERS_VERSION = "0.35.0.dev0"
PASSTHROUGH_COMPONENTS = ("vae", "text_encoder", "tokenizer")
WAN22_COMPONENT_MODULES: dict[str, list[str]] = {
    "vae": ["diffusers", "AutoencoderKLWan"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
}
COMPONENT_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "text_encoder": ("config.json", "model.safetensors.index.json"),
    "tokenizer": ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"),
}

HIDDEN_SIZE = 3072
FFN_DIM = 14336
NUM_LAYERS = 30
IN_CHANNELS = 48
OUT_CHANNELS = 48
TEXT_DIM = 4096
FREQ_DIM = 256
PATCH_SIZE = (1, 2, 2)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    pipeline_class_name: str
    hf_repo: str
    revision: str
    published_filename: str
    published_bytes: int
    published_sha256: str
    subject_ref_memory_max_refs: int
    causal: bool
    source_candidates: tuple[str, ...]


VARIANTS: dict[str, VariantSpec] = {
    "base_first_person": VariantSpec(
        name="base_first_person",
        pipeline_class_name="MatrixGame35BaseFirstPersonPipeline",
        hf_repo="RiemannDynamics/Matrix-Game-3.5-Base",
        revision="c3b0c9c541b7754a78b5e2199e9587e003668de9",
        published_filename="first-person.safetensors",
        published_bytes=10_000_464_984,
        published_sha256="3d758de69f545c835ad115f50b75719e682a83c18acdf219e6c720c5f3da5ea8",
        subject_ref_memory_max_refs=2,
        causal=False,
        source_candidates=(
            "base/first-person.safetensors",
            "Matrix-Game-3.5-Base/first-person.safetensors",
            "first-person.safetensors",
        ),
    ),
    "base_third_person": VariantSpec(
        name="base_third_person",
        pipeline_class_name="MatrixGame35BaseThirdPersonPipeline",
        hf_repo="RiemannDynamics/Matrix-Game-3.5-Base",
        revision="c3b0c9c541b7754a78b5e2199e9587e003668de9",
        published_filename="third-person.safetensors",
        published_bytes=10_000_477_272,
        published_sha256="3388cf355148355ce216ce18a44bd304574f7eaa8c636fb14c4cbd0b47d777cf",
        subject_ref_memory_max_refs=4,
        causal=False,
        source_candidates=(
            "base/third-person.safetensors",
            "Matrix-Game-3.5-Base/third-person.safetensors",
            "third-person.safetensors",
        ),
    ),
    "distilled_first_person": VariantSpec(
        name="distilled_first_person",
        pipeline_class_name="MatrixGame35DistilledFirstPersonPipeline",
        hf_repo="RiemannDynamics/Matrix-Game-3.5-Distilled",
        revision="0b38ca0b0dda2bb994c570e183ad36d1acd53be2",
        published_filename="first-person.safetensors",
        published_bytes=9_999_659_704,
        published_sha256="de476e7fc0bdd756aafb101a2b80040f65b3ad62dafea109e299aafa599b8094",
        subject_ref_memory_max_refs=0,
        causal=True,
        source_candidates=(
            "distilled/first-person.safetensors",
            "Matrix-Game-3.5-Distilled/first-person.safetensors",
            "distilled-first-person.safetensors",
        ),
    ),
}


def build_expected_official_shapes(spec: VariantSpec) -> dict[str, tuple[int, ...]]:
    """Build the exact released raw-DiT state surface without allocating it."""

    shapes: dict[str, tuple[int, ...]] = {
        "patch_embedding.weight": (HIDDEN_SIZE, IN_CHANNELS, *PATCH_SIZE),
        "patch_embedding.bias": (HIDDEN_SIZE, ),
        "text_embedding.0.weight": (HIDDEN_SIZE, TEXT_DIM),
        "text_embedding.0.bias": (HIDDEN_SIZE, ),
        "text_embedding.2.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        "text_embedding.2.bias": (HIDDEN_SIZE, ),
        "time_embedding.0.weight": (HIDDEN_SIZE, FREQ_DIM),
        "time_embedding.0.bias": (HIDDEN_SIZE, ),
        "time_embedding.2.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        "time_embedding.2.bias": (HIDDEN_SIZE, ),
        "time_projection.1.weight": (6 * HIDDEN_SIZE, HIDDEN_SIZE),
        "time_projection.1.bias": (6 * HIDDEN_SIZE, ),
        "head.modulation": (1, 2, HIDDEN_SIZE),
        "head.head.weight": (OUT_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2], HIDDEN_SIZE),
        "head.head.bias": (OUT_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2], ),
    }

    for block_index in range(NUM_LAYERS):
        prefix = f"blocks.{block_index}"
        for attention in ("self_attn", "cross_attn"):
            for projection in ("q", "k", "v", "o"):
                shapes[f"{prefix}.{attention}.{projection}.weight"] = (HIDDEN_SIZE, HIDDEN_SIZE)
                shapes[f"{prefix}.{attention}.{projection}.bias"] = (HIDDEN_SIZE, )
            shapes[f"{prefix}.{attention}.norm_q.weight"] = (HIDDEN_SIZE, )
            shapes[f"{prefix}.{attention}.norm_k.weight"] = (HIDDEN_SIZE, )

        shapes[f"{prefix}.ffn.0.weight"] = (FFN_DIM, HIDDEN_SIZE)
        shapes[f"{prefix}.ffn.0.bias"] = (FFN_DIM, )
        shapes[f"{prefix}.ffn.2.weight"] = (HIDDEN_SIZE, FFN_DIM)
        shapes[f"{prefix}.ffn.2.bias"] = (HIDDEN_SIZE, )
        shapes[f"{prefix}.norm3.weight"] = (HIDDEN_SIZE, )
        shapes[f"{prefix}.norm3.bias"] = (HIDDEN_SIZE, )
        shapes[f"{prefix}.modulation"] = (1, 6, HIDDEN_SIZE)

    max_refs = spec.subject_ref_memory_max_refs
    if max_refs:
        shapes.update({
            "subject_ref_index_embedding": (max_refs, HIDDEN_SIZE),
            "subject_ref_type_embedding": (1, HIDDEN_SIZE),
            "subject_ref_local_h_embedding": (64, HIDDEN_SIZE),
            "subject_ref_local_w_embedding": (64, HIDDEN_SIZE),
        })

    expected_count = 829 if max_refs else 825
    if len(shapes) != expected_count:
        raise AssertionError(f"Internal Matrix-Game 3.5 contract has {len(shapes)} keys, expected {expected_count}.")
    return shapes


def _namespace_for_key(key: str) -> str:
    for prefix in OFFICIAL_COMPAT_PREFIXES:
        if key.startswith(prefix):
            return prefix
    return ""


def normalize_namespace(keys: list[str]) -> tuple[str, dict[str, str]]:
    """Return the single checkpoint prefix and raw-to-unprefixed key map."""

    if not keys:
        raise ValueError("Checkpoint has no tensor keys.")
    namespaces = {_namespace_for_key(key) for key in keys}
    if len(namespaces) != 1:
        rendered = [namespace or "<unprefixed>" for namespace in sorted(namespaces)]
        raise ValueError(f"Checkpoint mixes incompatible key namespaces: {rendered}.")
    prefix = namespaces.pop()
    normalized: dict[str, str] = {}
    for raw_key in keys:
        key = raw_key.removeprefix(prefix) if prefix else raw_key
        if key in normalized.values():
            raise ValueError(f"Checkpoint key normalization collision: {key}.")
        normalized[raw_key] = key
    return prefix, normalized


def _header(path: Path) -> tuple[list[str], dict[str, tuple[int, ...]], dict[str, str]]:
    keys: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        for key in keys:
            tensor_slice = checkpoint.get_slice(key)
            shapes[key] = tuple(tensor_slice.get_shape())
            dtypes[key] = str(tensor_slice.get_dtype())
    return keys, shapes, dtypes


def _mapped_key(key: str) -> str:
    mapping = MatrixGame35WanVideoConfig().arch_config.param_names_mapping
    mapped, merge_index, split_count = get_param_names_mapping(mapping)(key)
    if merge_index is not None or split_count is not None:
        raise ValueError(f"Matrix-Game 3.5 unexpectedly requires tensor fusion for {key}.")
    if mapped == key and key not in SUBJECT_REF_KEYS:
        raise ValueError(f"No Matrix-Game 3.5 parameter mapping for inference key: {key}.")
    return mapped


def validate_source(path: Path, spec: VariantSpec, verify_sha256: bool) -> tuple[str, dict[str, str]]:
    """Validate the complete checkpoint header before loading any tensor data."""

    if not path.is_file():
        raise FileNotFoundError(f"Matrix-Game 3.5 source checkpoint does not exist: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"Matrix-Game 3.5 source must be a .safetensors file: {path}")

    raw_keys, raw_shapes, raw_dtypes = _header(path)
    prefix, normalized_names = normalize_namespace(raw_keys)
    normalized_shapes = {normalized_names[key]: shape for key, shape in raw_shapes.items()}
    normalized_dtypes = {normalized_names[key]: dtype for key, dtype in raw_dtypes.items()}
    expected_shapes = build_expected_official_shapes(spec)

    missing = sorted(set(expected_shapes) - set(normalized_shapes))
    unexpected = sorted(set(normalized_shapes) - set(expected_shapes))
    if missing or unexpected:
        raise ValueError(
            f"{spec.name} key contract mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}."
        )
    shape_mismatches = {
        key: (normalized_shapes[key], expected_shapes[key])
        for key in expected_shapes
        if normalized_shapes[key] != expected_shapes[key]
    }
    if shape_mismatches:
        first = next(iter(shape_mismatches.items()))
        raise ValueError(
            f"{spec.name} shape contract mismatch at {first[0]}: "
            f"got {first[1][0]}, expected {first[1][1]}."
        )
    bad_dtypes = {key: dtype for key, dtype in normalized_dtypes.items() if dtype != "BF16"}
    if bad_dtypes:
        first = next(iter(bad_dtypes.items()))
        raise ValueError(f"{spec.name} must be entirely BF16; {first[0]} is {first[1]}.")

    mapped: dict[str, str] = {}
    target_keys: set[str] = set()
    for raw_key, normalized_key in normalized_names.items():
        target_key = _mapped_key(normalized_key)
        if target_key in target_keys:
            raise ValueError(f"Matrix-Game 3.5 parameter mapping collision: {target_key}.")
        mapped[raw_key] = target_key
        target_keys.add(target_key)
    if len(mapped) != len(expected_shapes):
        raise AssertionError("Matrix-Game 3.5 conversion would drop inference tensors.")

    if verify_sha256:
        size = path.stat().st_size
        if size != spec.published_bytes:
            raise ValueError(f"{spec.name} byte size is {size}, expected {spec.published_bytes}.")
        digest = _sha256(path)
        if digest != spec.published_sha256:
            raise ValueError(f"{spec.name} SHA-256 is {digest}, expected {spec.published_sha256}.")
    return prefix, mapped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_transformer_config(spec: VariantSpec) -> dict[str, object]:
    config: dict[str, object] = {
        "_class_name": "MatrixGame35Transformer3DModel",
        "_diffusers_version": "0.33.1",
        "patch_size": list(PATCH_SIZE),
        "text_len": 512,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "text_dim": TEXT_DIM,
        "freq_dim": FREQ_DIM,
        "ffn_dim": FFN_DIM,
        "num_layers": NUM_LAYERS,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "use_prope": True,
        "prope_attention_interval": 1,
        "prope_camera_layout": "full",
        "prope_disable_native_rope": False,
        "subject_ref_memory_max_refs": spec.subject_ref_memory_max_refs,
        "causal": spec.causal,
        "causal_chunk_size": 3,
        "causal_window_size": 21,
    }
    validate_transformer_config(config)
    return config


def validate_transformer_config(payload: Mapping[str, object]) -> None:
    config = MatrixGame35WanVideoConfig()
    arch_fields = {field.name for field in fields(config.arch_config)}
    metadata_fields = {"_class_name", "_diffusers_version"}
    unknown = sorted(set(payload) - arch_fields - metadata_fields)
    if unknown:
        raise ValueError(f"Unknown Matrix-Game 3.5 transformer config fields: {unknown}.")
    if payload.get("_class_name") != "MatrixGame35Transformer3DModel":
        raise ValueError("Matrix-Game 3.5 transformer config has the wrong _class_name.")
    config.update_model_arch({key: value for key, value in payload.items() if key in arch_fields})
    for key, expected in payload.items():
        if key in metadata_fields:
            continue
        actual = getattr(config.arch_config, key)
        comparable_actual = list(actual) if isinstance(actual, tuple) else actual
        if comparable_actual != expected:
            raise ValueError(f"Matrix-Game 3.5 config round-trip mismatch for {key}: {actual!r} != {expected!r}.")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON component file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON component file must contain an object: {path}")
    return payload


def validate_component_source(source: Path) -> None:
    """Validate the pinned Wan2.2 passthrough surface before creating output."""

    if not source.is_dir():
        raise FileNotFoundError(f"Wan2.2 component source directory does not exist: {source}")
    model_index_path = source / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError(f"Missing pinned Wan2.2 component file: {model_index_path}")
    model_index = _read_json_object(model_index_path)
    expected_metadata = {"_diffusers_version": WAN22_DIFFUSERS_VERSION}
    for key, expected in expected_metadata.items():
        if model_index.get(key) != expected:
            raise ValueError(
                f"Pinned Wan2.2 model_index.json has {key}={model_index.get(key)!r}, expected {expected!r}."
            )
    for component, expected in WAN22_COMPONENT_MODULES.items():
        if model_index.get(component) != expected:
            raise ValueError(
                f"Pinned Wan2.2 model_index.json has {component}={model_index.get(component)!r}, "
                f"expected {expected!r}."
            )

    missing = [
        str(source / component / filename)
        for component, filenames in COMPONENT_REQUIRED_FILES.items()
        for filename in filenames
        if not (source / component / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing pinned Wan2.2 component files: {missing}.")

    _read_json_object(source / "vae" / "config.json")
    _read_json_object(source / "text_encoder" / "config.json")
    text_index = _read_json_object(source / "text_encoder" / "model.safetensors.index.json")
    weight_map = text_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Pinned Wan2.2 text encoder index must contain a non-empty weight_map.")
    shard_values = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in shard_values):
        raise ValueError("Pinned Wan2.2 text encoder index contains an invalid shard name.")
    shard_names = set(shard_values)
    for shard_name in shard_names:
        shard_path = Path(shard_name)
        if shard_path.is_absolute() or ".." in shard_path.parts:
            raise ValueError(f"Pinned Wan2.2 text encoder index contains an unsafe shard path: {shard_name!r}.")
        resolved = source / "text_encoder" / shard_path
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing pinned Wan2.2 text encoder shard: {resolved}")


def build_model_index(spec: VariantSpec) -> dict[str, object]:
    return {
        "_class_name": spec.pipeline_class_name,
        "_diffusers_version": WAN22_DIFFUSERS_VERSION,
        "transformer": ["diffusers", "MatrixGame35Transformer3DModel"],
        **WAN22_COMPONENT_MODULES,
    }


def _materialize_components(source: Path, destination: Path, *, symlink: bool) -> None:
    for component in PASSTHROUGH_COMPONENTS:
        component_source = source / component
        component_destination = destination / component
        if symlink:
            component_destination.symlink_to(component_source.resolve(), target_is_directory=True)
        else:
            shutil.copytree(component_source, component_destination)


def convert_variant(
    source: Path,
    destination_root: Path,
    spec: VariantSpec,
    *,
    verify_sha256: bool = False,
    overwrite: bool = False,
    component_source: Path | None = None,
    component_revision: str | None = None,
    symlink_components: bool = False,
) -> Path:
    """Convert one preflighted variant into an atomic transformer directory."""

    prefix, key_mapping = validate_source(source, spec, verify_sha256)
    if component_source is None and component_revision is not None:
        raise ValueError("component_revision requires component_source.")
    if symlink_components and component_source is None:
        raise ValueError("symlink_components requires component_source.")
    if component_source is not None:
        if component_revision != WAN22_COMPONENT_REVISION:
            raise ValueError(
                "component_revision must pin the supported Wan2.2 Diffusers snapshot "
                f"{WAN22_COMPONENT_REVISION}, got {component_revision!r}."
            )
        validate_component_source(component_source)
    destination_root.mkdir(parents=True, exist_ok=True)
    final_dir = destination_root / spec.name
    if final_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing converted variant: {final_dir}")

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{spec.name}-", dir=destination_root))
    try:
        transformer_dir = staging_dir / "transformer"
        transformer_dir.mkdir()
        raw_state = load_file(source, device="cpu")
        converted: OrderedDict[str, torch.Tensor] = OrderedDict()
        for raw_key in sorted(raw_state):
            target_key = key_mapping[raw_key]
            if target_key in converted:
                raise ValueError(f"Matrix-Game 3.5 parameter mapping collision: {target_key}.")
            converted[target_key] = raw_state[raw_key]
        if len(converted) != len(raw_state):
            raise AssertionError("Matrix-Game 3.5 conversion dropped inference tensors.")

        weight_path = transformer_dir / "diffusion_pytorch_model.safetensors"
        save_file(converted, weight_path, metadata={"format": "pt"})
        _write_json(transformer_dir / "config.json", build_transformer_config(spec))

        output_keys, output_shapes, output_dtypes = _header(weight_path)
        expected_target_shapes = {
            key_mapping[raw_key]: tuple(tensor.shape) for raw_key, tensor in raw_state.items()
        }
        if set(output_keys) != set(expected_target_shapes):
            raise AssertionError("Saved Matrix-Game 3.5 transformer keys changed during serialization.")
        if output_shapes != expected_target_shapes:
            raise AssertionError("Saved Matrix-Game 3.5 transformer shapes changed during serialization.")
        if set(output_dtypes.values()) != {"BF16"}:
            raise AssertionError("Saved Matrix-Game 3.5 transformer did not preserve BF16.")

        if component_source is not None:
            _materialize_components(component_source, staging_dir, symlink=symlink_components)
            _write_json(staging_dir / "model_index.json", build_model_index(spec))

        manifest: dict[str, object] = {
            "format_version": 1,
            "variant": spec.name,
            "source_layout": "raw_official",
            "source_namespace": prefix or "unprefixed",
            "source_filename": source.name,
            "hf_repo": spec.hf_repo,
            "revision": spec.revision,
            "published_bytes": spec.published_bytes,
            "published_sha256": spec.published_sha256,
            "sha256_verified": verify_sha256,
            "input_key_count": len(raw_state),
            "output_key_count": len(converted),
            "skipped_keys": list(SKIP_PATTERNS),
        }
        if component_source is not None:
            manifest["component_source"] = {
                "repo": WAN22_COMPONENT_REPO,
                "revision": component_revision,
                "materialization": "symlink" if symlink_components else "copy",
            }
        _write_json(staging_dir / "conversion.json", manifest)

        del converted
        del raw_state
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging_dir, final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return final_dir


def resolve_source(source: Path, spec: VariantSpec) -> Path:
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Matrix-Game 3.5 source path does not exist: {source}")
    matches = [source / relative for relative in spec.source_candidates if (source / relative).is_file()]
    if not matches:
        expected = ", ".join(spec.source_candidates)
        raise FileNotFoundError(f"No {spec.name} checkpoint under {source}; expected one of: {expected}.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {spec.name} sources under {source}: {[str(path) for path in matches]}.")
    return matches[0]


def _selected_variants(name: str) -> list[VariantSpec]:
    if name == "all":
        return list(VARIANTS.values())
    return [VARIANTS[name]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Local checkpoint file or root containing all variants.",
    )
    parser.add_argument("--dst", type=Path, required=True, help="Local output root for variant transformer folders.")
    parser.add_argument("--variant", choices=("all", *VARIANTS), default="all")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Require the exact published byte size and SHA-256 before conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output variant only after the new conversion has serialized successfully.",
    )
    parser.add_argument(
        "--component-source",
        type=Path,
        help=(
            f"Optional local materialization of {WAN22_COMPONENT_REPO}@{WAN22_COMPONENT_REVISION}; "
            "copies vae/text_encoder/tokenizer and writes model_index.json."
        ),
    )
    parser.add_argument(
        "--component-revision",
        help=f"Required with --component-source; must equal the pinned revision {WAN22_COMPONENT_REVISION}.",
    )
    parser.add_argument(
        "--symlink-components",
        action="store_true",
        help="Symlink passthrough components from --component-source for local development instead of copying.",
    )
    args = parser.parse_args()

    if args.component_source is None and args.component_revision is not None:
        parser.error("--component-revision requires --component-source.")
    if args.component_source is not None and args.component_revision is None:
        parser.error("--component-source requires --component-revision.")
    if args.symlink_components and args.component_source is None:
        parser.error("--symlink-components requires --component-source.")

    specs = _selected_variants(args.variant)
    if args.src.is_file() and len(specs) != 1:
        parser.error("A file --src requires one explicit --variant, not --variant all.")

    resolved = [(spec, resolve_source(args.src, spec)) for spec in specs]
    for spec, source in resolved:
        validate_source(source, spec, args.verify_sha256)
    for spec, source in resolved:
        output = convert_variant(
            source,
            args.dst,
            spec,
            verify_sha256=args.verify_sha256,
            overwrite=args.overwrite,
            component_source=args.component_source,
            component_revision=args.component_revision,
            symlink_components=args.symlink_components,
        )
        print(f"converted {spec.name}: {source} -> {output}")


if __name__ == "__main__":
    main()
