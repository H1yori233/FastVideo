# SPDX-License-Identifier: Apache-2.0
"""Real-weight FLUX.3 DiT parity on one bf16 CUDA GPU (24 GB minimum).

Scope: strict production TransformerLoader using official Modular Diffusers weights.
"""
from __future__ import annotations

import gc
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(os.getenv("FLUX3_OFFICIAL_REF_DIR", ROOT / "flux-3-image-klein"))
WEIGHTS = Path(os.getenv("FLUX3_MODEL_DIR", ROOT / "official_weights/flux3/flux-3-image-klein"))
PARITY_SCOPE = "production_loader"

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29683")


def _inputs(packed_class, editing: bool):
    generator = torch.Generator().manual_seed(143)
    lengths = {"image": [16, 12], "image_cond": [0, 4] if editing else [0, 0], "text": [6, 3]}
    channels = {"image": 128, "image_cond": 128, "text": 20480}
    data, ids, timesteps, indices = {}, {}, {}, {}
    for mode, sizes in lengths.items():
        count = sum(sizes)
        data[mode] = torch.randn(count, channels[mode], generator=generator).to("cuda", torch.bfloat16)
        if mode == "image":
            data[mode] = data[mode].float()
        ids[mode] = torch.zeros(count, 4, device="cuda")
        ids[mode][:, 0] = torch.arange(count, device="cuda") % 2
        ids[mode][:, 1] = torch.arange(count, device="cuda") // 4
        ids[mode][:, 2] = torch.arange(count, device="cuda") % 4
        ids[mode][:, 3] = 10 if mode == "image_cond" else 0
        timesteps[mode] = torch.full((count,), 0.7 if mode == "image" else 0, device="cuda", dtype=torch.bfloat16)
        indices[mode] = [(0, sizes[0]), (sizes[0], count)]
    return packed_class(data, ids, timesteps, indices), torch.tensor([0.4, 0.2], device="cuda")


def _run(model, packed_class):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return [model(*_inputs(packed_class, editing)).float().cpu() for editing in (False, True)]


@pytest.fixture(scope="module")
def official_outputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not (REFERENCE / "src/flux3/dit.py").exists() or not (WEIGHTS / "model.safetensors").exists():
        pytest.skip("FLUX.3 official checkout or weights are absent")
    sys.path.insert(0, str(REFERENCE / "src"))
    official = importlib.import_module("flux3.dit")
    with torch.device("meta"):
        reference = official.DiT(flow_map=True)
    reference.load_state_dict(load_file(str(WEIGHTS / "model.safetensors"), device="cuda"), strict=True, assign=True)
    reference.eval()
    expected = _run(reference, official.Packed)
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    yield expected
    from fastvideo.distributed import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


def _load_production():
    from fastvideo.configs.models.dits.flux3 import Flux3Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.models.loader.component_loader import TransformerLoader

    component = WEIGHTS / "transformer"
    if not (component / "config.json").exists():
        pytest.skip("FLUX.3 official transformer is absent")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=Flux3Config(), dit_precision="bf16"),
        model_paths={},
        override_transformer_cls_name=None,
        init_weights_from_safetensors=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        dit_cpu_offload=False,
        pin_cpu_memory=False,
        use_fsdp_inference=False,
        training_mode=False,
        enable_torch_compile=False,
        torch_compile_kwargs={},
        inference_torch_compile=False,
        VSA_tile_size=None,
        inference_mode=True,
        dit_layerwise_offload=False,
    )
    return TransformerLoader().load(str(component), args)


def test_flux3_real_weight_parity(official_outputs):
    from fastvideo.models.dits import flux3 as native

    model = _load_production()
    actual = _run(model.eval(), native.Packed)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for mode, output, target in zip(("t2i", "editing"), actual, official_outputs, strict=True):
        diff = (output - target).abs()
        drift = diff.mean() / target.abs().mean().clamp_min(1e-8)
        print(f"production/{mode}: abs_mean={target.abs().mean():.8f} diff_max={diff.max():.8f} "
              f"diff_mean={diff.mean():.8f} relative_mean={drift:.8f}")
        torch.testing.assert_close(output, target, atol=1e-2, rtol=1e-2)
        assert drift < 0.01
