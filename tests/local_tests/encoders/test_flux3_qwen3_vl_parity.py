# SPDX-License-Identifier: Apache-2.0
"""Production-loader parity for FLUX.3's text-only Qwen3-VL conditioner."""

import gc
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

PARITY_SCOPE = "production_loader"
ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = Path(os.getenv("FLUX3_OFFICIAL_REF_DIR", ROOT / "flux-3-image-klein"))
MODEL = Path(os.getenv("FLUX3_MODEL_DIR", ROOT / "official_weights/flux3/flux-3-image-klein"))
ENCODER = MODEL / "text_encoder"


def test_official_nested_config_selects_native_text_encoder():
    from fastvideo.configs.models.encoders.flux3 import Flux3TextEncoderConfig

    source = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {"hidden_size": 32, "num_hidden_layers": 1, "rope_theta": 2048.0,
                        "rope_scaling": {"mrope_section": [2, 3, 3]}},
        "vision_config": {"hidden_size": 999},
    }
    original = deepcopy(source)
    config = Flux3TextEncoderConfig()
    config.update_model_arch(source)
    assert config.architectures == ["Flux3TextEncoder"]
    assert config.hidden_size == 32 and config.num_hidden_layers == 1
    assert config.rope_theta == 2048.0 and config.mrope_section == (2, 3, 3)
    assert source == original


@pytest.fixture
def tiny_encoder_checkpoint():
    from fastvideo.configs.models.encoders.flux3 import Flux3TextEncoderConfig
    from fastvideo.models.encoders.flux3 import Flux3TextEncoder

    config = Flux3TextEncoderConfig()
    config.update_model_arch({"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                              "vocab_size": 64, "mrope_section": (2, 3, 3)})
    model = Flux3TextEncoder(config)
    native = dict(model.named_parameters())
    official = [("model.language_model." + name, torch.zeros_like(value)) for name, value in native.items()]
    official.append(("model.visual.patch_embed.proj.weight", torch.zeros(1)))
    return model, official


def test_official_weight_prefix_mapping_keeps_strict_language_loading(tiny_encoder_checkpoint):
    model, official = tiny_encoder_checkpoint
    native = dict(model.named_parameters())
    assert model.load_weights(official) == set(native)
    with pytest.raises(ValueError, match="Missing FLUX.3 text encoder tensors"):
        model.load_weights(official[1:])
    with pytest.raises(ValueError, match="Unexpected FLUX.3 text encoder tensor"):
        model.load_weights(official + [("model.language_model.unknown.weight", torch.zeros(1))])


@pytest.mark.parametrize("source_prefix", ["model.language_model.", ""])
def test_text_encoder_rejects_duplicate_mapped_weights(tiny_encoder_checkpoint, source_prefix):
    model, official = tiny_encoder_checkpoint
    name, parameter = next(iter(model.named_parameters()))
    duplicate = (source_prefix + name, torch.ones_like(parameter))
    with pytest.raises(ValueError, match="Duplicate FLUX.3 text encoder tensor"):
        model.load_weights(official + [duplicate])
    assert torch.count_nonzero(parameter) == 0


def test_flux3_qwen3_vl_parity():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is required")
    if not (OFFICIAL / "src/flux3/text_encoder.py").exists() or not (ENCODER / "config.json").exists():
        pytest.skip("FLUX.3 official source and encoder weights are required")

    from fastvideo.configs.models.encoders.flux3 import Flux3TextEncoderConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.models.loader.component_loader import TextEncoderLoader
    from fastvideo.models.encoders.flux3 import Flux3TextEncoder
    from transformers import AutoTokenizer

    sys.path.insert(0, str(OFFICIAL / "src"))
    from flux3.text_encoder import TextEncoder

    prompts = ["A red fox on fresh snow.", "雨后的上海街道，霓虹灯倒映在水面。\nA cinematic photograph.", "detailed landscape " * 90]
    if os.getenv("FLUX3_ENCODER_TEST_TRUNCATION", "1") == "1":
        prompts.append("landscape " * 5000)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29684")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        reference = TextEncoder(str(ENCODER), device="cuda")
        expected = [tuple(value.cpu() for value in reference.encode(prompt)) for prompt in prompts]
        del reference
        gc.collect()
        torch.cuda.empty_cache()

        args = SimpleNamespace(
            pipeline_config=SimpleNamespace(text_encoder_configs=(Flux3TextEncoderConfig(),),
                                            text_encoder_precisions=("bf16",)),
            text_encoder_cpu_offload=False,
            override_text_encoder_quant=None,
            override_text_encoder_safetensors=None,
            pin_cpu_memory=False,
            disable_offload_on_unified_memory=lambda *args, **kwargs: False,
        )
        model = TextEncoderLoader().load(str(ENCODER), args)
        assert isinstance(model, Flux3TextEncoder)
        tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        for index, (prompt, (tokens, ids)) in enumerate(zip(prompts, expected, strict=True)):
            actual_tokens, actual_ids = model.encode(prompt, tokenizer)
            actual_tokens, actual_ids = actual_tokens.cpu(), actual_ids.cpu()
            torch.testing.assert_close(actual_ids, ids, atol=0, rtol=0)
            torch.testing.assert_close(actual_tokens, tokens, atol=0, rtol=0)
            difference = (actual_tokens.float() - tokens.float()).abs()
            print(f"case={index} length={len(ids)} max_abs={difference.max():.8f} mean_abs={difference.mean():.8f}")
            assert len(ids) <= 4096
            assert len(ids) == 4096 or len(ids) % 80 == 0
        del model
    finally:
        cleanup_dist_env_and_memory()
