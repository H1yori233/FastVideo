# SPDX-License-Identifier: Apache-2.0
"""Real-weight PUP language parity; one 32 GB CUDA GPU, BF16, Torch fallbacks.

Scope: implementation_subcomponent. The PUP wrapper owns production-loader parity.
"""

import gc
import json
import os
from pathlib import Path

import pytest
import torch
from accelerate import init_empty_weights
from safetensors import safe_open

PARITY_SCOPE = "implementation_subcomponent"
ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = Path(os.getenv("FLUX3_PUP_WEIGHTS_DIR", ROOT / "official_weights/flux3/pup"))
PREFIX = "model.language_model."
PREFILL_LENGTH = 12
DECODE_LENGTHS = (1, 2, 1)


def _native_class():
    from fastvideo.models.encoders.flux3_pup.text import TextModel
    return TextModel


def _load_model(model_class, config, device):
    native = model_class.__module__.startswith("fastvideo.")
    context = torch.device("meta") if native else init_empty_weights(include_buffers=False)
    with context:
        model = model_class(config)
    with safe_open(WEIGHTS / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        state = {key.removeprefix(PREFIX): checkpoint.get_tensor(key) for key in checkpoint.keys()
                 if key.startswith(PREFIX)}
    assert state, "PUP checkpoint contains no language-model weights"
    model.load_state_dict(state, strict=True, assign=True)
    del state
    if native:
        model.materialize_non_persistent_buffers(device)
    return model.to(device=device).eval()


def _inputs(case, device, config):
    batch_size = 2 if case == "left_padding" else 1
    prefill_length = 129 if case == "long_text" else PREFILL_LENGTH
    generator = torch.Generator().manual_seed(718)
    tokens = torch.randint(100, config["text_config"]["vocab_size"] - 256,
                           (batch_size, prefill_length + sum(DECODE_LENGTHS)), generator=generator, device="cpu")
    mask = torch.ones_like(tokens)
    positions = None
    if case == "left_padding":
        tokens[1, :3] = config["pad_token_id"]
        mask[1, :3] = 0
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
    elif case == "multimodal_positions":
        positions = torch.arange(tokens.shape[1]).view(1, 1, -1).repeat(4, batch_size, 1)
        positions[1, :, 4:8] = 4
        positions[2, :, 4:8] = torch.tensor([4, 4, 5, 5])
        positions[3, :, 4:8] = torch.tensor([4, 5, 4, 5])
        positions[1:, :, 8:] -= 2
    return tokens.to(device), mask.to(device), None if positions is None else positions.to(device)


@torch.inference_mode()
def _run_sequence(model, case, device, config):
    tokens, mask, positions = _inputs(case, device, config)
    outputs = []
    cache = None
    start = 0
    prefill_length = 129 if case == "long_text" else PREFILL_LENGTH
    for step, length in enumerate((prefill_length, *DECODE_LENGTHS)):
        end = start + length
        result = model(input_ids=tokens[:, start:end],
                       attention_mask=mask[:, :end],
                       position_ids=None if positions is None else positions[..., start:end],
                       past_key_values=cache,
                       use_cache=True)
        cache = result.past_key_values
        assert cache is not None, f"{case}: missing cache at step {step}"
        assert cache.get_seq_length() == end
        hidden = result.last_hidden_state
        assert hidden.shape == (tokens.shape[0], end - start, config["text_config"]["hidden_size"])
        assert hidden.dtype == torch.bfloat16
        assert torch.isfinite(hidden).all()
        outputs.append(hidden.float().cpu())
        start = end
    return outputs


def test_flux3_pup_text_prefill_and_cached_decode_parity(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("A 32 GB CUDA GPU is required")
    if not (WEIGHTS / "config.json").exists() or not (WEIGHTS / "model.safetensors").exists():
        pytest.skip("FLUX.3 PUP config and real weights are required")
    native_class = _native_class()

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as official

    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        monkeypatch.setattr(official, name, None)

    config = json.loads((WEIGHTS / "config.json").read_text())
    official_config = Qwen3_5TextConfig(**config["text_config"])
    official_config._attn_implementation = "sdpa"
    device = torch.device("cuda:0")
    cases = ("text", "left_padding", "multimodal_positions", "long_text")
    reference = _load_model(official.Qwen3_5TextModel, official_config, device)
    try:
        expected = {case: _run_sequence(reference, case, device, config) for case in cases}
    finally:
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    native = _load_model(native_class, config["text_config"], device)
    try:
        for case in cases:
            actual = _run_sequence(native, case, device, config)
            for step, (result, target) in enumerate(zip(actual, expected[case], strict=True)):
                difference = (result - target).abs()
                print(f"case={case} step={step} max_abs={difference.max():.8f} "
                      f"mean_abs={difference.mean():.8f}")
                torch.testing.assert_close(result, target, atol=0, rtol=0)
    finally:
        del native
        gc.collect()
        torch.cuda.empty_cache()
