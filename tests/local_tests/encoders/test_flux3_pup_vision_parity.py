# SPDX-License-Identifier: Apache-2.0
"""Native PUP vision parity with real BF16 weights; CPU or one CUDA GPU."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

PARITY_SCOPE = "implementation_subcomponent"
ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = Path(os.getenv("FLUX3_PUP_WEIGHTS_DIR", ROOT / "official_weights/flux3/pup"))


@pytest.fixture(scope="module")
def models():
    if not (WEIGHTS_DIR / "model.safetensors").is_file():
        pytest.skip("Local FLUX.3 PUP weights are required")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding
    from fastvideo.models.encoders.flux3_pup.vision import Flux3PupVisionModel

    config = json.loads((WEIGHTS_DIR / "config.json").read_text())["vision_config"]
    official_config = Qwen3_5VisionConfig(**config)
    official_config._attn_implementation = "sdpa"
    device = torch.device(os.getenv("FLUX3_PUP_VISION_DEVICE", "cpu"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    with torch.device("meta"):
        official = Qwen3_5VisionModel(official_config)
        native = Flux3PupVisionModel(SimpleNamespace(**config))
    with safe_open(str(WEIGHTS_DIR / "model.safetensors"), framework="pt", device=str(device)) as file:
        state = {key.removeprefix("model.visual."): file.get_tensor(key) for key in file.keys()
                 if key.startswith("model.visual.")}
    official.load_state_dict(state, strict=True, assign=True)
    native.load_state_dict(state, strict=True, assign=True)
    head_dim = config["hidden_size"] // config["num_heads"]
    official.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2).to(device)
    native.to(device)
    native.materialize_non_persistent_buffers(device)
    yield official.eval(), native.eval(), config, device
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("grid", [[[1, 4, 4]], [[1, 4, 6]], [[1, 4, 4], [1, 2, 6]]])
@torch.inference_mode()
def test_flux3_pup_vision_real_weights(models, grid):
    official, native, config, device = models
    grid_thw = torch.tensor(grid, device=device)
    count = sum(t * h * w for t, h, w in grid)
    channels = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"]**2
    pixels = torch.randn(count, channels, generator=torch.Generator().manual_seed(37)).to(device, torch.bfloat16)
    expected = official(pixels, grid_thw)
    actual = native(pixels, grid_thw)
    for name in ("last_hidden_state", "pooler_output"):
        reference, result = getattr(expected, name), getattr(actual, name)
        delta = (result.float() - reference.float()).abs()
        print(f"grid={grid} {name}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g}")
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_flux3_pup_vision_rotary_construction(models):
    from fastvideo.models.encoders.flux3_pup.vision import Flux3PupVisionRotaryEmbedding

    official, _, config, device = models
    with torch.device(device):
        native = Flux3PupVisionRotaryEmbedding(config["hidden_size"] // config["num_heads"] // 2)
    torch.testing.assert_close(native.inv_freq, official.rotary_pos_emb.inv_freq, atol=0, rtol=0)
