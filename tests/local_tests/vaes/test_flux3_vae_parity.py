# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 VAE FP32 parity, including channel-major packing and production loading.

Runs on CPU by default; set FLUX3_PARITY_DEVICE=cuda for a single CUDA GPU.
"""
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file

PARITY_SCOPE = "both"
REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("FLUX3_OFFICIAL_REF_DIR", REPO_ROOT / "flux-3-image-klein"))
WEIGHTS = Path(os.getenv("FLUX3_VAE_WEIGHTS", REPO_ROOT / "official_weights/flux3/flux-3-image-klein/ae.safetensors"))
MODEL_DIR = Path(os.getenv("FLUX3_MODEL_DIR", REPO_ROOT / "official_weights/flux3/flux-3-image-klein"))
VAE_DIR = MODEL_DIR / "vae"


@pytest.fixture(scope="module")
def models():
    if not WEIGHTS.is_file() or not OFFICIAL_REF_DIR.is_dir():
        pytest.skip("Local FLUX.3 reference source and VAE weights are required")
    sys.path.insert(0, str(OFFICIAL_REF_DIR / "src"))
    official_module = importlib.import_module("flux3.autoencoder")
    from fastvideo.configs.models.vaes.flux3 import Flux3VAEConfig
    from fastvideo.models.vaes.flux3 import Flux3AutoEncoder

    device = torch.device(os.getenv("FLUX3_PARITY_DEVICE", "cpu"))
    previous_threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(4)
    if not (VAE_DIR / "diffusion_pytorch_model.safetensors").is_file():
        pytest.skip(f"Official Modular Diffusers VAE weights missing: {VAE_DIR}")
    weights = load_file(str(WEIGHTS))
    official = official_module.AutoEncoder()
    config = Flux3VAEConfig()
    config.update_model_arch(json.loads((VAE_DIR / "config.json").read_text()))
    native = Flux3AutoEncoder(config)
    official.load_state_dict(weights, strict=True)
    modular_weights = load_file(str(VAE_DIR / "diffusion_pytorch_model.safetensors"))
    native.load_state_dict(modular_weights, strict=True)
    for name, tensor in native.state_dict().items():
        # The published Diffusers checkpoint resets this training-only counter.
        if name == "bn.num_batches_tracked":
            torch.testing.assert_close(tensor, modular_weights[name], atol=0, rtol=0)
            continue
        torch.testing.assert_close(tensor, weights[name], atol=0, rtol=0)
    yield official.to(device).eval(), native.to(device).eval(), device
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("height,width", [(32, 48), (64, 64)])
def test_flux3_vae_encode_decode_parity(models, height, width):
    official, native, device = models
    generator = torch.Generator().manual_seed(42)
    image = torch.rand(3, height, width, generator=generator).mul(2).sub(1).to(device)
    expected_tokens, expected_ids = official.encode(image)
    actual_tokens, actual_ids = native.encode(image)
    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_tokens, expected_tokens, atol=1e-5, rtol=1e-5)
    expected_image = official.decode(expected_tokens, expected_ids)
    actual_image = native.decode(expected_tokens, expected_ids)
    torch.testing.assert_close(actual_image, expected_image, atol=1e-5, rtol=1e-5)
    print(f"{height}x{width}: encode max_abs={(actual_tokens - expected_tokens).abs().max().item():.6g}; "
          f"decode max_abs={(actual_image - expected_image).abs().max().item():.6g}")


def test_flux3_vae_production_loader_parity(models):
    if not (VAE_DIR / "config.json").is_file():
        pytest.skip(f"Official VAE missing: {VAE_DIR}")
    from fastvideo.configs.models.vaes.flux3 import Flux3VAEConfig
    from fastvideo.models.vaes.flux3 import Flux3AutoEncoder

    official, _, device = models
    native = Flux3AutoEncoder.from_pretrained(VAE_DIR, config=Flux3VAEConfig(), dtype=torch.float32, device=device)
    image = torch.linspace(-1, 1, 3 * 32 * 48, device=device).reshape(3, 32, 48)
    expected_tokens, expected_ids = official.encode(image)
    actual_tokens, actual_ids = native.encode(image)
    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_tokens, expected_tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(native.decode(expected_tokens, expected_ids), official.decode(expected_tokens, expected_ids),
                               atol=1e-5, rtol=1e-5)


def test_flux3_vae_rejects_missing_batchnorm_counter(models):
    _, native, _ = models
    state = dict(native.state_dict())
    del state["bn.num_batches_tracked"]
    with pytest.raises(RuntimeError, match="bn.num_batches_tracked"):
        native.load_state_dict(state, strict=True)
