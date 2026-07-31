# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the Matrix-Game 3.5 checkpoint converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts.checkpoint_conversion import matrixgame35_to_diffusers as converter
from tests.local_tests.matrixgame35._upstream import load_upstream_transformer


def _write_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, path)


def _tiny_contract(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[int, ...]]:
    shapes = {
        "patch_embedding.bias": (2, ),
        "blocks.0.self_attn.q.bias": (2, ),
        "head.modulation": (1, 2, 2),
    }
    monkeypatch.setattr(converter, "build_expected_official_shapes", lambda _spec: shapes)
    return shapes


def _write_component_source(path: Path) -> Path:
    modules = {
        "_class_name": "RobotWMV0CausalPipeline",
        "_diffusers_version": "0.35.0.dev0",
        **converter.WAN22_COMPONENT_MODULES,
    }
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(json.dumps(modules), encoding="utf-8")
    for component in converter.PASSTHROUGH_COMPONENTS:
        (path / component).mkdir()

    (path / "vae" / "config.json").write_text("{}", encoding="utf-8")
    (path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    (path / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    text_index = {"weight_map": {"encoder.block.0.weight": "model-00001-of-00001.safetensors"}}
    (path / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps(text_index),
        encoding="utf-8",
    )
    (path / "text_encoder" / "model-00001-of-00001.safetensors").write_bytes(b"text")
    for filename in converter.COMPONENT_REQUIRED_FILES["tokenizer"]:
        (path / "tokenizer" / filename).write_bytes(b"tokenizer")
    return path


def test_all_released_variant_surfaces_are_exact() -> None:
    first = converter.build_expected_official_shapes(converter.VARIANTS["base_first_person"])
    third = converter.build_expected_official_shapes(converter.VARIANTS["base_third_person"])
    distilled = converter.build_expected_official_shapes(converter.VARIANTS["distilled_first_person"])

    assert len(first) == 829
    assert len(third) == 829
    assert len(distilled) == 825
    assert first["subject_ref_index_embedding"] == (2, 3072)
    assert third["subject_ref_index_embedding"] == (4, 3072)
    assert first["subject_ref_type_embedding"] == (1, 3072)
    assert first["subject_ref_local_h_embedding"] == (64, 3072)
    assert first["subject_ref_local_w_embedding"] == (64, 3072)
    assert not set(converter.SUBJECT_REF_KEYS) & set(distilled)
    assert set(first) - set(converter.SUBJECT_REF_KEYS) == set(distilled)
    assert set(third) - set(converter.SUBJECT_REF_KEYS) == set(distilled)


def test_shape_contract_matches_pinned_official_meta_models() -> None:
    reference_dir = Path(__file__).resolve().parents[3] / "Matrix-Game-3.5"
    if not (reference_dir / "diffsynth" / "models" / "wan_video_dit.py").is_file():
        pytest.skip("Pinned Matrix-Game 3.5 source clone is absent.")
    wan = load_upstream_transformer(reference_dir).wan_video_dit
    common_kwargs = {
        "has_image_input": False,
        "patch_size": converter.PATCH_SIZE,
        "in_dim": converter.IN_CHANNELS,
        "dim": converter.HIDDEN_SIZE,
        "ffn_dim": converter.FFN_DIM,
        "freq_dim": converter.FREQ_DIM,
        "text_dim": converter.TEXT_DIM,
        "out_dim": converter.OUT_CHANNELS,
        "num_heads": 24,
        "num_layers": converter.NUM_LAYERS,
        "eps": 1e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_prope": True,
        "prope_camera_layout": "full",
    }

    for spec in converter.VARIANTS.values():
        max_refs = spec.subject_ref_memory_max_refs
        with torch.device("meta"):
            model = wan.WanModel(
                **common_kwargs,
                subject_ref_memory_enabled=max_refs > 0,
                subject_ref_memory_max_refs=max_refs or 2,
            )
        actual = {key: tuple(tensor.shape) for key, tensor in model.state_dict().items()}
        assert actual == converter.build_expected_official_shapes(spec)


def test_mapping_uses_native_config_surface() -> None:
    assert converter._mapped_key("patch_embedding.weight") == "patch_embedding.proj.weight"
    assert converter._mapped_key("blocks.4.self_attn.q.bias") == "blocks.4.to_q.bias"
    assert converter._mapped_key("blocks.7.cross_attn.o.weight") == "blocks.7.attn2.to_out.weight"
    assert converter._mapped_key("blocks.12.ffn.0.weight") == "blocks.12.ffn.fc_in.weight"
    assert converter._mapped_key("subject_ref_index_embedding") == "subject_ref_index_embedding"
    with pytest.raises(ValueError, match="No Matrix-Game 3.5 parameter mapping"):
        converter._mapped_key("unknown_inference_weight")


@pytest.mark.parametrize(
    ("keys", "expected_prefix"),
    [
        (["patch_embedding.bias", "head.modulation"], ""),
        (["pipe.dit.patch_embedding.bias", "pipe.dit.head.modulation"], "pipe.dit."),
        (["dit.patch_embedding.bias", "dit.head.modulation"], "dit."),
    ],
)
def test_namespace_is_all_or_nothing(keys: list[str], expected_prefix: str) -> None:
    prefix, mapping = converter.normalize_namespace(keys)
    assert prefix == expected_prefix
    assert set(mapping.values()) == {"patch_embedding.bias", "head.modulation"}


def test_namespace_rejects_mixed_prefixes() -> None:
    with pytest.raises(ValueError, match="mixes incompatible key namespaces"):
        converter.normalize_namespace([
            "patch_embedding.bias",
            "pipe.dit.head.modulation",
        ])
    with pytest.raises(ValueError, match="mixes incompatible key namespaces"):
        converter.normalize_namespace([
            "pipe.dit.patch_embedding.bias",
            "dit.head.modulation",
        ])


def test_convert_variant_preserves_bf16_and_writes_variant_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / "source.safetensors"
    _write_checkpoint(
        source,
        {
            f"pipe.dit.{key}": torch.arange(torch.tensor(shape).prod().item(), dtype=torch.bfloat16).reshape(shape)
            for key, shape in shapes.items()
        },
    )

    spec = converter.VARIANTS["base_third_person"]
    output = converter.convert_variant(source, tmp_path / "converted", spec)
    transformer = output / "transformer"
    config = json.loads((transformer / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "conversion.json").read_text(encoding="utf-8"))

    assert config["_class_name"] == "MatrixGame35Transformer3DModel"
    assert config["subject_ref_memory_max_refs"] == 4
    assert config["causal"] is False
    assert manifest["variant"] == "base_third_person"
    assert manifest["source_namespace"] == "pipe.dit."
    assert manifest["input_key_count"] == len(shapes)
    assert manifest["output_key_count"] == len(shapes)
    assert manifest["skipped_keys"] == []
    assert not (output / "model_index.json").exists()
    assert not set(converter.PASSTHROUGH_COMPONENTS) & {path.name for path in output.iterdir()}

    with safe_open(transformer / "diffusion_pytorch_model.safetensors", framework="pt") as checkpoint:
        assert set(checkpoint.keys()) == {
            "patch_embedding.proj.bias",
            "blocks.0.to_q.bias",
            "scale_shift_table",
        }
        assert {str(checkpoint.get_slice(key).get_dtype()) for key in checkpoint.keys()} == {"BF16"}


def test_convert_variant_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / "source.safetensors"
    _write_checkpoint(source, {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()})
    spec = converter.VARIANTS["distilled_first_person"]

    converter.convert_variant(source, tmp_path / "converted", spec)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        converter.convert_variant(source, tmp_path / "converted", spec)
    output = converter.convert_variant(source, tmp_path / "converted", spec, overwrite=True)
    config = json.loads((output / "transformer" / "config.json").read_text(encoding="utf-8"))
    assert config["causal"] is True
    assert config["subject_ref_memory_max_refs"] == 0


def test_source_validation_rejects_dtype_and_unmapped_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.safetensors"
    spec = converter.VARIANTS["distilled_first_person"]

    monkeypatch.setattr(
        converter,
        "build_expected_official_shapes",
        lambda _spec: {"patch_embedding.bias": (2, )},
    )
    _write_checkpoint(source, {"patch_embedding.bias": torch.zeros(2, dtype=torch.float32)})
    with pytest.raises(ValueError, match="must be entirely BF16"):
        converter.validate_source(source, spec, verify_sha256=False)

    monkeypatch.setattr(
        converter,
        "build_expected_official_shapes",
        lambda _spec: {"unknown_inference_weight": (2, )},
    )
    _write_checkpoint(source, {"unknown_inference_weight": torch.zeros(2, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="No Matrix-Game 3.5 parameter mapping"):
        converter.validate_source(source, spec, verify_sha256=False)


def test_published_digest_check_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / "source.safetensors"
    _write_checkpoint(source, {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()})
    spec = converter.VARIANTS["base_first_person"]

    converter.validate_source(source, spec, verify_sha256=False)
    with pytest.raises(ValueError, match="byte size"):
        converter.validate_source(source, spec, verify_sha256=True)


def test_source_root_resolution_rejects_ambiguity(tmp_path: Path) -> None:
    spec = converter.VARIANTS["base_first_person"]
    for relative in spec.source_candidates[:2]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with pytest.raises(ValueError, match="Ambiguous base_first_person"):
        converter.resolve_source(tmp_path, spec)


@pytest.mark.parametrize(
    ("variant", "pipeline_class"),
    [
        ("base_first_person", "MatrixGame35BaseFirstPersonPipeline"),
        ("base_third_person", "MatrixGame35BaseThirdPersonPipeline"),
        ("distilled_first_person", "MatrixGame35DistilledFirstPersonPipeline"),
    ],
)
def test_complete_root_copies_components_and_writes_variant_model_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    pipeline_class: str,
) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / f"{variant}.safetensors"
    _write_checkpoint(source, {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()})
    component_source = _write_component_source(tmp_path / f"wan-{variant}")

    output = converter.convert_variant(
        source,
        tmp_path / "converted",
        converter.VARIANTS[variant],
        component_source=component_source,
        component_revision=converter.WAN22_COMPONENT_REVISION,
    )
    model_index = json.loads((output / "model_index.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "conversion.json").read_text(encoding="utf-8"))

    assert model_index["_class_name"] == pipeline_class
    assert set(model_index) == {
        "_class_name",
        "_diffusers_version",
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
    }
    assert model_index["transformer"] == ["diffusers", "MatrixGame35Transformer3DModel"]
    assert {key: model_index[key] for key in converter.PASSTHROUGH_COMPONENTS} == (
        converter.WAN22_COMPONENT_MODULES
    )
    for component in converter.PASSTHROUGH_COMPONENTS:
        assert (output / component).is_dir()
        assert not (output / component).is_symlink()
    assert manifest["component_source"] == {
        "repo": converter.WAN22_COMPONENT_REPO,
        "revision": converter.WAN22_COMPONENT_REVISION,
        "materialization": "copy",
    }


def test_complete_root_can_symlink_components_for_local_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / "source.safetensors"
    _write_checkpoint(source, {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()})
    component_source = _write_component_source(tmp_path / "wan")

    output = converter.convert_variant(
        source,
        tmp_path / "converted",
        converter.VARIANTS["base_first_person"],
        component_source=component_source,
        component_revision=converter.WAN22_COMPONENT_REVISION,
        symlink_components=True,
    )

    for component in converter.PASSTHROUGH_COMPONENTS:
        component_path = output / component
        assert component_path.is_symlink()
        assert component_path.resolve() == (component_source / component).resolve()
    manifest = json.loads((output / "conversion.json").read_text(encoding="utf-8"))
    assert manifest["component_source"]["materialization"] == "symlink"


def test_missing_component_is_rejected_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / "source.safetensors"
    _write_checkpoint(source, {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()})
    component_source = _write_component_source(tmp_path / "wan")
    (component_source / "tokenizer" / "spiece.model").unlink()
    destination = tmp_path / "converted"

    with pytest.raises(FileNotFoundError, match="spiece.model"):
        converter.convert_variant(
            source,
            destination,
            converter.VARIANTS["base_first_person"],
            component_source=component_source,
            component_revision=converter.WAN22_COMPONENT_REVISION,
        )

    assert not destination.exists()


def test_component_source_requires_the_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes = _tiny_contract(monkeypatch)
    source = tmp_path / "source.safetensors"
    _write_checkpoint(source, {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()})
    component_source = _write_component_source(tmp_path / "wan")

    with pytest.raises(ValueError, match="component_revision must pin"):
        converter.convert_variant(
            source,
            tmp_path / "converted",
            converter.VARIANTS["base_first_person"],
            component_source=component_source,
        )
