# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 base transformer parity scaffold.

Coverage scope: both. For each released base checkpoint, the official side
strict-loads the pinned safetensors into upstream ``WanModel`` with Warped PRoPE
enabled in every block. The FastVideo side loads the matching converted
transformer through ``TransformerLoader`` into
``MatrixGame35Transformer3DModel``.

The real-weight CUDA path is intentionally sequential so the two roughly 10 GB
BF16 models do not coexist on a 40 GB GPU. A scaffold skip is not parity
evidence; activation requires a non-skip pass with the pinned checkpoint.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
import torch
from torch.testing import assert_close

from fastvideo.forward_context import set_forward_context
from tests.local_tests.matrixgame35._upstream import (
    PINNED_OFFICIAL_REVISION,
    UpstreamTransformerModules,
    load_upstream_transformer,
)


os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29535")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_SCOPE = "both"

OFFICIAL_HF_REPO = "RiemannDynamics/Matrix-Game-3.5-Base"
OFFICIAL_HF_REVISION = "c3b0c9c541b7754a78b5e2199e9587e003668de9"
OFFICIAL_COMPAT_STATE_PREFIXES = ("pipe.dit.", "dit.")

OFFICIAL_REF_DIR = Path(
    os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5")
)
BASE_WEIGHTS_DIR = Path(
    os.getenv(
        "MATRIXGAME35_BASE_WEIGHTS_DIR",
        REPO_ROOT / "official_weights" / "matrixgame35" / "base",
    )
)


@dataclass(frozen=True)
class BaseVariantSpec:
    name: str
    official_weight_name: str
    official_weight_sha256: str
    subject_ref_memory_max_refs: int
    official_weight_env: str
    converted_transformer_env: str

    @property
    def official_weight_path(self) -> Path:
        return Path(
            os.getenv(
                self.official_weight_env,
                BASE_WEIGHTS_DIR / self.official_weight_name,
            )
        )

    @property
    def converted_transformer_dir(self) -> Path:
        return Path(
            os.getenv(
                self.converted_transformer_env,
                REPO_ROOT
                / "converted_weights"
                / "matrixgame35"
                / self.name
                / "transformer",
            )
        )


BASE_VARIANTS = (
    BaseVariantSpec(
        name="base_first_person",
        official_weight_name="first-person.safetensors",
        official_weight_sha256="3d758de69f545c835ad115f50b75719e682a83c18acdf219e6c720c5f3da5ea8",
        subject_ref_memory_max_refs=2,
        official_weight_env="MATRIXGAME35_FIRST_PERSON_WEIGHTS",
        converted_transformer_env="MATRIXGAME35_CONVERTED_TRANSFORMER_DIR",
    ),
    BaseVariantSpec(
        name="base_third_person",
        official_weight_name="third-person.safetensors",
        official_weight_sha256="3388cf355148355ce216ce18a44bd304574f7eaa8c636fb14c4cbd0b47d777cf",
        subject_ref_memory_max_refs=4,
        official_weight_env="MATRIXGAME35_THIRD_PERSON_WEIGHTS",
        converted_transformer_env="MATRIXGAME35_THIRD_PERSON_CONVERTED_TRANSFORMER_DIR",
    ),
)

FASTVIDEO_CONFIG_MODULE = "fastvideo.configs.models.dits.matrixgame35"
FASTVIDEO_CONFIG_CLASS = "MatrixGame35WanVideoConfig"
FASTVIDEO_MODEL_MODULE = "fastvideo.models.dits.matrixgame35"
FASTVIDEO_MODEL_CLASS = "MatrixGame35Transformer3DModel"

OFFICIAL_MODEL_KWARGS: dict[str, Any] = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "use_prope": True,
    "prope_disable_native_rope": False,
    "prope_disable_t_rope": False,
    "prope_camera_layout": "full",
    "subject_ref_memory_enabled": True,
}

EXPECTED_FASTVIDEO_CONFIG: dict[str, Any] = {
    "_class_name": FASTVIDEO_MODEL_CLASS,
    "patch_size": [1, 2, 2],
    "in_channels": 48,
    "out_channels": 48,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "ffn_dim": 14336,
    "num_layers": 30,
    "text_dim": 4096,
    "freq_dim": 256,
    "use_prope": True,
    "prope_attention_interval": 1,
    "prope_camera_layout": "full",
    "prope_disable_native_rope": False,
    "causal": False,
}


def _skip_if_reference_missing() -> None:
    source = OFFICIAL_REF_DIR / "diffsynth" / "models" / "wan_video_dit.py"
    if not source.is_file():
        pytest.skip(
            "Pinned Matrix-Game 3.5 reference is absent; set "
            f"MATRIXGAME35_OFFICIAL_REF_DIR to commit {PINNED_OFFICIAL_REVISION}."
        )


def _skip_if_official_weights_missing(spec: BaseVariantSpec) -> None:
    if not spec.official_weight_path.is_file():
        pytest.skip(
            f"Official transformer weight is absent: {spec.official_weight_path}. "
            f"Stage {OFFICIAL_HF_REPO}@{OFFICIAL_HF_REVISION}/{spec.official_weight_name} "
            f"or set {spec.official_weight_env}."
        )


def _skip_if_converted_weights_missing(spec: BaseVariantSpec) -> None:
    if not spec.converted_transformer_dir.is_dir() or not any(
        spec.converted_transformer_dir.glob("*.safetensors")
    ):
        pytest.skip(
            "Converted Matrix-Game 3.5 transformer is absent; set "
            f"{spec.converted_transformer_env} "
            f"(expected {spec.converted_transformer_dir})."
        )


@lru_cache(maxsize=None)
def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_official_checkpoint_identity(spec: BaseVariantSpec) -> None:
    actual_sha256 = _checkpoint_sha256(spec.official_weight_path)
    assert actual_sha256 == spec.official_weight_sha256, (
        f"Unexpected SHA-256 for {spec.name} checkpoint {spec.official_weight_path}: "
        f"{actual_sha256}; expected {spec.official_weight_sha256}"
    )


def _import_fastvideo_contract_or_skip():
    try:
        config_module = importlib.import_module(FASTVIDEO_CONFIG_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == FASTVIDEO_CONFIG_MODULE:
            pytest.skip(f"Planned FastVideo config module is not implemented: {FASTVIDEO_CONFIG_MODULE}")
        raise
    try:
        model_module = importlib.import_module(FASTVIDEO_MODEL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == FASTVIDEO_MODEL_MODULE:
            pytest.skip(f"Planned FastVideo model module is not implemented: {FASTVIDEO_MODEL_MODULE}")
        raise

    config_class = getattr(config_module, FASTVIDEO_CONFIG_CLASS, None)
    model_class = getattr(model_module, FASTVIDEO_MODEL_CLASS, None)
    if config_class is None:
        pytest.skip(
            f"Planned FastVideo config class is not implemented: "
            f"{FASTVIDEO_CONFIG_MODULE}.{FASTVIDEO_CONFIG_CLASS}"
        )
    if model_class is None:
        pytest.skip(
            f"Planned FastVideo model class is not implemented: "
            f"{FASTVIDEO_MODEL_MODULE}.{FASTVIDEO_MODEL_CLASS}"
        )
    return config_class, model_class


def _build_official_meta_model(
    modules: UpstreamTransformerModules,
    spec: BaseVariantSpec,
) -> torch.nn.Module:
    with torch.device("meta"):
        model = modules.wan_video_dit.WanModel(
            **OFFICIAL_MODEL_KWARGS,
            subject_ref_memory_max_refs=spec.subject_ref_memory_max_refs,
        )
    assert model.use_prope is True
    assert model.seperated_timestep is True
    assert model.subject_ref_memory_enabled is True
    assert tuple(model.subject_ref_index_embedding.shape) == (
        spec.subject_ref_memory_max_refs,
        3072,
    )
    assert len(model.blocks) == 30
    assert all(block.use_prope and block.self_attn.use_prope for block in model.blocks)
    return model


def _official_state_prefix(raw_keys: list[str]) -> str:
    """Detect an optional legacy wrapper prefix without accepting mixed keys."""

    assert raw_keys, "Official checkpoint has no tensor keys"
    for prefix in OFFICIAL_COMPAT_STATE_PREFIXES:
        prefixed = [key.startswith(prefix) for key in raw_keys]
        if any(prefixed):
            assert all(prefixed), (
                f"Official checkpoint mixes {prefix!r}-prefixed and unprefixed keys"
            )
            return prefix
    return ""


def _checkpoint_shapes(path: Path) -> tuple[dict[str, tuple[int, ...]], set[str]]:
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: set[str] = set()
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        raw_keys = list(checkpoint.keys())
        prefix = _official_state_prefix(raw_keys)
        for raw_key in raw_keys:
            key = raw_key.removeprefix(prefix) if prefix else raw_key
            assert key not in shapes, f"Official key normalization collision: {key}"
            tensor_slice = checkpoint.get_slice(raw_key)
            shapes[key] = tuple(tensor_slice.get_shape())
            dtypes.add(str(tensor_slice.get_dtype()))
    return shapes, dtypes


def _load_official_model(
    modules: UpstreamTransformerModules,
    device: torch.device,
    spec: BaseVariantSpec,
) -> torch.nn.Module:
    model = _build_official_meta_model(modules, spec)
    raw_state = safetensors_load_file(str(spec.official_weight_path), device="cpu")
    assert raw_state, f"Official checkpoint is empty: {spec.official_weight_path}"
    prefix = _official_state_prefix(list(raw_state))
    state = {
        (key.removeprefix(prefix) if prefix else key): tensor
        for key, tensor in raw_state.items()
    }
    assert len(state) == len(raw_state), "Official key normalization produced collisions"
    del raw_state

    expected = model.state_dict()
    assert set(state) == set(expected), (
        f"Official strict key mismatch: missing={sorted(set(expected) - set(state))[:8]} "
        f"unexpected={sorted(set(state) - set(expected))[:8]}"
    )
    shape_mismatches = {
        key: (tuple(state[key].shape), tuple(expected[key].shape))
        for key in state
        if tuple(state[key].shape) != tuple(expected[key].shape)
    }
    assert not shape_mismatches, f"Official checkpoint shape mismatch: {shape_mismatches}"
    assert {tensor.dtype for tensor in state.values()} == {torch.bfloat16}

    incompatible = model.load_state_dict(state, strict=True, assign=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    del state

    # ``freqs`` is a plain upstream tensor list, not checkpoint state. Meta
    # construction therefore needs the real deterministic table restored.
    model.freqs = modules.wan_video_dit.precompute_freqs_cis_3d(128)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    assert not any(parameter.is_meta for parameter in model.parameters())
    return model


def _load_fastvideo_model(
    device: torch.device,
    spec: BaseVariantSpec,
) -> torch.nn.Module:
    config_class, model_class = _import_fastvideo_contract_or_skip()
    transformer_dir = spec.converted_transformer_dir
    config_path = transformer_dir / "config.json"
    if not config_path.is_file():
        pytest.fail(f"Converted transformer directory has no config.json: {config_path}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_FASTVIDEO_CONFIG.items():
        assert config_payload.get(key) == expected, (
            f"Unexpected converted Matrix-Game config {key}={config_payload.get(key)!r}; "
            f"expected {expected!r}"
        )
    assert config_payload.get("subject_ref_memory_max_refs") == spec.subject_ref_memory_max_refs, (
        "Unexpected converted Matrix-Game subject-reference row count: "
        f"{config_payload.get('subject_ref_memory_max_refs')!r}; "
        f"expected {spec.subject_ref_memory_max_refs!r} for {spec.name}"
    )

    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader

    args = FastVideoArgs(
        model_path=str(transformer_dir),
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=config_class(),
            dit_precision="bf16",
        ),
    )
    model = TransformerLoader().load(str(transformer_dir), args)
    assert isinstance(model, model_class)
    assert args.model_paths["transformer"] == str(transformer_dir)
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert next(model.parameters()).device == device
    assert next(model.parameters()).dtype == torch.bfloat16
    assert model.subject_ref_memory_max_refs == spec.subject_ref_memory_max_refs
    assert tuple(model.subject_ref_index_embedding.shape) == (
        spec.subject_ref_memory_max_refs,
        3072,
    )
    return model.eval()


def _make_camera_info(
    modules: UpstreamTransformerModules,
    *,
    latent_frames: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build the official `(w2c, (P, P_T, P_inv))` four-camera contract."""

    batch = 1
    subframes = 4
    w2c = torch.eye(4, dtype=torch.float32).reshape(1, 1, 1, 4, 4).repeat(
        batch, latent_frames, subframes, 1, 1
    )
    for frame in range(latent_frames):
        for subframe in range(subframes):
            angle = 0.01 * (4 * frame + subframe)
            cosine = torch.cos(torch.tensor(angle))
            sine = torch.sin(torch.tensor(angle))
            w2c[0, frame, subframe, 0, 0] = cosine
            w2c[0, frame, subframe, 0, 2] = sine
            w2c[0, frame, subframe, 2, 0] = -sine
            w2c[0, frame, subframe, 2, 2] = cosine
            w2c[0, frame, subframe, :3, 3] = torch.tensor(
                [0.02 * frame, -0.01 * subframe, 0.005 * (4 * frame + subframe)]
            )

    intrinsics = torch.zeros(
        batch, latent_frames, subframes, 3, 3, dtype=torch.float32
    )
    intrinsics[..., 0, 0] = 0.82
    intrinsics[..., 1, 1] = 0.91
    intrinsics[..., 0, 2] = 0.015
    intrinsics[..., 1, 2] = -0.02
    intrinsics[..., 2, 2] = 1.0

    prope = modules.prope_attention
    projection = torch.einsum(
        "...ij,...jk->...ik", prope.lift_k(intrinsics), w2c
    )
    projection_t = projection.transpose(-1, -2)
    projection_inv = torch.einsum(
        "...ij,...jk->...ik",
        prope.invert_se3(w2c),
        prope.lift_k(prope.invert_k(intrinsics)),
    )
    camera_info = (
        w2c.to(device=device, dtype=dtype),
        tuple(
            tensor.to(device=device, dtype=dtype)
            for tensor in (projection, projection_t, projection_inv)
        ),
    )
    assert camera_info[0].shape == (1, latent_frames, 4, 4, 4)
    assert all(
        tensor.shape == (1, latent_frames, 4, 4, 4)
        for tensor in camera_info[1]
    )
    return camera_info


def _make_inputs(
    modules: UpstreamTransformerModules,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    latent_frames = 2
    latent_height = 4
    latent_width = 4
    tokens_per_frame = (latent_height // 2) * (latent_width // 2)
    hidden_states = torch.randn(
        1,
        48,
        latent_frames,
        latent_height,
        latent_width,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(
        1,
        512,
        4096,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    timestep = torch.tensor(
        [[0.0] * tokens_per_frame + [500.0] * tokens_per_frame],
        device=device,
        dtype=torch.float32,
    )
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "camera_info": _make_camera_info(
            modules,
            latent_frames=latent_frames,
            device=device,
            dtype=dtype,
        ),
    }


def _build_official_freqs(model: torch.nn.Module, f: int, h: int, w: int, device: torch.device):
    return torch.cat(
        [
            model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(device)


def _run_official(
    modules: UpstreamTransformerModules,
    model: torch.nn.Module,
    inputs: dict[str, Any],
) -> torch.Tensor:
    """Execute the real upstream non-mosaic transformer path with PRoPE."""

    wan = modules.wan_video_dit
    hidden_states = inputs["hidden_states"]
    timestep = inputs["timestep"]
    context = inputs["encoder_hidden_states"]
    camera_info = inputs["camera_info"]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = model.patchify(hidden_states)
        f, h, w = x.shape[2:]
        x = wan.rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        assert timestep.shape == (x.shape[0], x.shape[1])
        t = model.time_embedding(
            wan.sinusoidal_embedding_1d(model.freq_dim, timestep.reshape(-1)).unsqueeze(0)
        )
        t_mod = model.time_projection(t).unflatten(2, (6, model.dim))
        context = model.text_embedding(context)
        freqs = _build_official_freqs(model, f, h, w, x.device)
        for block in model.blocks:
            x = block(x, context, t_mod, freqs, camera_info=camera_info)
        x = model.head(x, t)
        output = model.unpatchify(x, (f, h, w))
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    return output.detach().float().cpu()


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("sample", output.get("x"))
    elif hasattr(output, "sample"):
        output = output.sample
    elif isinstance(output, tuple):
        output = output[0]
    assert torch.is_tensor(output), f"Transformer output is not a tensor: {type(output)}"
    return output


def _run_fastvideo(model: torch.nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        output = model(**inputs)
    output = _extract_tensor(output)
    assert output.shape == inputs["hidden_states"].shape
    assert torch.isfinite(output).all()
    return output.detach().float().cpu()


def test_matrixgame35_official_key_prefix_contract() -> None:
    """Released keys are raw; two historical wrapper prefixes stay compatible."""

    assert _official_state_prefix(["blocks.0.modulation", "patch_embedding.weight"]) == ""
    assert _official_state_prefix(["pipe.dit.blocks.0.modulation", "pipe.dit.patch_embedding.weight"]) == "pipe.dit."
    assert _official_state_prefix(["dit.blocks.0.modulation", "dit.patch_embedding.weight"]) == "dit."
    with pytest.raises(AssertionError, match="mixes"):
        _official_state_prefix(["blocks.0.modulation", "pipe.dit.patch_embedding.weight"])


def test_matrixgame35_upstream_narrow_import_executes_real_prope_cpu() -> None:
    """The pinned transformer imports without optional umbrella dependencies."""

    _skip_if_reference_missing()
    modules_before = set(sys.modules)
    modules = load_upstream_transformer(OFFICIAL_REF_DIR)
    newly_loaded = set(sys.modules) - modules_before
    assert not any(
        name == "modelscope" or name.startswith("modelscope.")
        for name in newly_loaded
    )
    assert modules.wan_video_dit.WanModel.__module__ == modules.wan_video_dit.__name__

    torch.manual_seed(17)
    attention = modules.wan_video_dit.SelfAttention(
        dim=128,
        num_heads=1,
        use_prope=True,
        prope_camera_layout="full",
    ).eval()
    hidden_states = torch.randn(1, 2, 128)
    freqs = torch.ones(2, 1, 64, dtype=torch.complex128)
    projection = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, 4, 1, 1)
    camera_info = (projection, (projection, projection.transpose(-1, -2), projection))
    with torch.inference_mode():
        output = attention(hidden_states, freqs, camera_info=camera_info)
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    assert output.abs().max() > 0


@pytest.mark.parametrize("spec", BASE_VARIANTS, ids=lambda spec: spec.name)
def test_matrixgame35_base_official_key_surface(spec: BaseVariantSpec) -> None:
    """Compare each released base header with its exact upstream model surface."""

    _skip_if_reference_missing()
    _skip_if_official_weights_missing(spec)
    _assert_official_checkpoint_identity(spec)
    modules = load_upstream_transformer(OFFICIAL_REF_DIR)
    model = _build_official_meta_model(modules, spec)
    expected_shapes = {
        key: tuple(tensor.shape) for key, tensor in model.state_dict().items()
    }
    checkpoint_shapes, checkpoint_dtypes = _checkpoint_shapes(spec.official_weight_path)
    assert checkpoint_shapes == expected_shapes
    assert checkpoint_dtypes == {"BF16"}
    assert "subject_ref_index_embedding" in checkpoint_shapes
    assert checkpoint_shapes["subject_ref_index_embedding"] == (
        spec.subject_ref_memory_max_refs,
        3072,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Real-weight Matrix-Game 3.5 transformer parity requires CUDA.",
)
@pytest.mark.parametrize("spec", BASE_VARIANTS, ids=lambda spec: spec.name)
def test_matrixgame35_base_transformer_parity(spec: BaseVariantSpec) -> None:
    """Strict-load and compare each real official/FastVideo base transformer."""

    assert torch.cuda.is_bf16_supported(), "Matrix-Game 3.5 parity requires BF16-capable CUDA hardware"
    _skip_if_reference_missing()
    _skip_if_official_weights_missing(spec)
    _assert_official_checkpoint_identity(spec)
    _skip_if_converted_weights_missing(spec)
    _import_fastvideo_contract_or_skip()

    modules = load_upstream_transformer(OFFICIAL_REF_DIR)
    device = torch.device("cuda:0")
    inputs = _make_inputs(modules, device, torch.bfloat16)

    official = _load_official_model(modules, device, spec)
    official_output = _run_official(modules, official, inputs)
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_model(device, spec)
    fastvideo_output = _run_fastvideo(fastvideo, inputs)
    del fastvideo
    gc.collect()
    torch.cuda.empty_cache()

    assert fastvideo_output.shape == official_output.shape
    difference = (fastvideo_output - official_output).abs()
    official_abs_mean = official_output.abs().mean().clamp_min(1e-6)
    fastvideo_abs_mean = fastvideo_output.abs().mean()
    abs_mean_drift = (fastvideo_abs_mean - official_abs_mean).abs() / official_abs_mean
    normalized_mean_error = difference.mean() / official_abs_mean
    cosine = torch.nn.functional.cosine_similarity(
        fastvideo_output.flatten(), official_output.flatten(), dim=0
    )
    print(
        f"Matrix-Game 3.5 {spec.name} transformer parity: "
        f"official_abs_mean={official_abs_mean.item():.6f} "
        f"fastvideo_abs_mean={fastvideo_abs_mean.item():.6f} "
        f"diff_max={difference.max().item():.6f} "
        f"diff_mean={difference.mean().item():.6f} "
        f"normalized_mean_error={normalized_mean_error.item():.6f} "
        f"abs_mean_drift={abs_mean_drift.item():.6f} "
        f"cosine={cosine.item():.6f}"
    )
    assert abs_mean_drift < 0.05
    assert normalized_mean_error < 0.05
    assert cosine > 0.99
    assert_close(fastvideo_output, official_output, atol=0.1, rtol=0.1)
