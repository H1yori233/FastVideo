# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 UMT5/tokenizer reuse contracts and parity.

Coverage scope: both. The CUDA path loads Matrix-Game's raw WanTextEncoder
weights and FastVideo's native UMT5 through the production text loader. The
tokenizer check compares the official fixed-length cleaning path directly.
"""

from __future__ import annotations

import gc
import inspect
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close
from transformers import AutoTokenizer, UMT5EncoderModel as TransformersUMT5EncoderModel

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.matrixgame35 import (
    make_matrixgame35_text_encoder_config,
    matrixgame35_postprocess_text,
    matrixgame35_preprocess_text,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TextEncoderLoader, TokenizerLoader
from fastvideo.models.loader.utils import set_default_torch_dtype
from tests.local_tests.matrixgame35._shared_upstream import (
    load_upstream_wan_text_encoder,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_SCOPE = "both"
OFFICIAL_REF_DIR = Path(
    os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5")
)
WAN22_RAW_DIR = Path(
    os.getenv(
        "MATRIXGAME35_WAN22_RAW_DIR",
        REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B",
    )
)
WAN22_DIFFUSERS_DIR = Path(
    os.getenv(
        "MATRIXGAME35_WAN22_DIFFUSERS_DIR",
        REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B-Diffusers",
    )
)
RAW_TEXT_ENCODER_PATH = WAN22_RAW_DIR / "models_t5_umt5-xxl-enc-bf16.pth"
FASTVIDEO_TEXT_ENCODER_DIR = WAN22_DIFFUSERS_DIR / "text_encoder"
TOKENIZER_DIR = Path(
    os.getenv(
        "MATRIXGAME35_TOKENIZER_DIR",
        WAN22_RAW_DIR / "google" / "umt5-xxl",
    )
)
DIFFUSERS_TOKENIZER_DIR = WAN22_DIFFUSERS_DIR / "tokenizer"


def _patch_single_process_text_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.layers.linear as fastvideo_linear
    import fastvideo.layers.vocab_parallel_embedding as fastvideo_embedding
    import fastvideo.models.encoders.t5 as fastvideo_t5

    for module in (fastvideo_t5, fastvideo_embedding, fastvideo_linear):
        if hasattr(module, "get_tp_rank"):
            monkeypatch.setattr(module, "get_tp_rank", lambda: 0)
        if hasattr(module, "get_tp_world_size"):
            monkeypatch.setattr(module, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(
        fastvideo_embedding,
        "tensor_model_parallel_all_reduce",
        lambda tensor: tensor,
    )


def _load_official_text_encoder(device: torch.device) -> torch.nn.Module:
    if not RAW_TEXT_ENCODER_PATH.is_file():
        pytest.skip(
            f"Raw Wan2.2 UMT5 weights are absent: {RAW_TEXT_ENCODER_PATH}"
        )
    module = load_upstream_wan_text_encoder(OFFICIAL_REF_DIR)
    with set_default_torch_dtype(torch.bfloat16), torch.device(device):
        model = module.WanTextEncoder()
    state = torch.load(
        RAW_TEXT_ENCODER_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    del state
    return model.eval().requires_grad_(False)


def _load_transformers_text_encoder(
    device: torch.device,
) -> TransformersUMT5EncoderModel:
    if not (FASTVIDEO_TEXT_ENCODER_DIR / "config.json").is_file():
        pytest.skip(
            f"Diffusers-style Wan2.2 UMT5 is absent: {FASTVIDEO_TEXT_ENCODER_DIR}"
        )
    return (
        TransformersUMT5EncoderModel.from_pretrained(
            FASTVIDEO_TEXT_ENCODER_DIR,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )


def _load_fastvideo_text_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> torch.nn.Module:
    if not (FASTVIDEO_TEXT_ENCODER_DIR / "config.json").is_file():
        pytest.skip(
            f"Diffusers-style Wan2.2 UMT5 is absent: {FASTVIDEO_TEXT_ENCODER_DIR}"
        )
    _patch_single_process_text_parallel(monkeypatch)
    config = make_matrixgame35_text_encoder_config()
    config._fsdp_shard_conditions = []
    pipeline_config = PipelineConfig(
        text_encoder_configs=(config,),
        text_encoder_precisions=("bf16",),
        preprocess_text_funcs=(matrixgame35_preprocess_text,),
        postprocess_text_funcs=(matrixgame35_postprocess_text,),
    )
    args = FastVideoArgs(
        model_path=str(WAN22_DIFFUSERS_DIR),
        pipeline_config=pipeline_config,
        pin_cpu_memory=False,
        text_encoder_cpu_offload=False,
    )
    return TextEncoderLoader().load(str(FASTVIDEO_TEXT_ENCODER_DIR), args)


def _official_tokens(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not TOKENIZER_DIR.is_dir():
        pytest.skip(f"Wan2.2 UMT5 tokenizer is absent: {TOKENIZER_DIR}")
    module = load_upstream_wan_text_encoder(OFFICIAL_REF_DIR)
    tokenizer = module.HuggingfaceTokenizer(
        name=str(TOKENIZER_DIR),
        seq_len=512,
        clean="whitespace",
    )
    return tokenizer(prompt, return_mask=True, add_special_tokens=True)


def _diffusers_tokenizer() -> AutoTokenizer:
    if not DIFFUSERS_TOKENIZER_DIR.is_dir():
        pytest.skip(
            f"Diffusers-style Wan2.2 tokenizer is absent: {DIFFUSERS_TOKENIZER_DIR}"
        )
    return AutoTokenizer.from_pretrained(
        DIFFUSERS_TOKENIZER_DIR,
        local_files_only=True,
    )


def _matrixgame35_pipeline_config() -> PipelineConfig:
    config = make_matrixgame35_text_encoder_config()
    return PipelineConfig(
        text_encoder_configs=(config,),
        text_encoder_precisions=("bf16",),
        preprocess_text_funcs=(matrixgame35_preprocess_text,),
        postprocess_text_funcs=(matrixgame35_postprocess_text,),
    )


def test_matrixgame35_text_config_matches_official_wan_umt5() -> None:
    config = make_matrixgame35_text_encoder_config()
    module = load_upstream_wan_text_encoder(OFFICIAL_REF_DIR)
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            module.WanTextEncoder.__init__
        ).parameters.items()
    }

    assert config.prefix == "umt5"
    assert config.vocab_size == defaults["vocab"] == 256384
    assert config.d_model == defaults["dim"] == 4096
    assert config.d_kv * config.num_heads == defaults["dim_attn"] == 4096
    assert config.d_ff == defaults["dim_ffn"] == 10240
    assert config.num_heads == defaults["num_heads"] == 64
    assert config.num_layers == defaults["num_layers"] == 24
    assert config.relative_attention_num_buckets == defaults["num_buckets"] == 32
    assert config.dropout_rate == defaults["dropout"] == 0.1
    assert defaults["shared_pos"] is False
    assert config.feed_forward_proj == "gated-gelu"
    assert config.is_gated_act is True
    assert config.dense_act_fn == "gelu_new"
    assert config.is_encoder_decoder is False
    assert config.text_len == 512
    assert config.tokenizer_kwargs == {
        "truncation": True,
        "max_length": 512,
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_tensors": "pt",
        "padding": "max_length",
    }


@pytest.mark.parametrize(
    "prompt",
    (
        "  Move\tforward\nnow  ",
        "Tom &amp; Jerry &amp;amp; friends",
        "Unicode\u00a0spacing",
    ),
)
def test_matrixgame35_text_cleaning_matches_official(prompt: str) -> None:
    module = load_upstream_wan_text_encoder(OFFICIAL_REF_DIR)
    expected = module.whitespace_clean(module.basic_clean(prompt))
    assert matrixgame35_preprocess_text(prompt) == expected


def test_matrixgame35_text_postprocess_zero_pads_to_512() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    output = matrixgame35_postprocess_text(
        BaseEncoderOutput(last_hidden_state=hidden, attention_mask=mask)
    )

    assert output.shape == (2, 512, 3)
    assert_close(output[0, :2], hidden[0, :2])
    assert_close(output[1, :3], hidden[1, :3])
    assert torch.count_nonzero(output[0, 2:]) == 0
    assert torch.count_nonzero(output[1, 3:]) == 0


def test_matrixgame35_tokenizer_matches_official_fixed_length_path() -> None:
    prompt = "  Turn &amp; move\n forward.  "
    official_ids, official_mask = _official_tokens(prompt)
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
    )
    config = make_matrixgame35_text_encoder_config()
    fastvideo_tokens = tokenizer(
        [matrixgame35_preprocess_text(prompt)],
        **config.tokenizer_kwargs,
    )

    assert torch.equal(fastvideo_tokens.input_ids, official_ids)
    assert torch.equal(fastvideo_tokens.attention_mask, official_mask)
    assert official_ids.shape == official_mask.shape == (1, 512)


def test_matrixgame35_diffusers_tokenizer_matches_production_path() -> None:
    """Compare the pinned HF tokenizer to FastVideo's production loader."""
    prompt = "  Turn &amp; move\n forward.  "
    reference = _diffusers_tokenizer()
    pipeline_config = _matrixgame35_pipeline_config()
    args = FastVideoArgs(
        model_path=str(WAN22_DIFFUSERS_DIR),
        pipeline_config=pipeline_config,
        pin_cpu_memory=False,
    )
    production = TokenizerLoader().load(str(DIFFUSERS_TOKENIZER_DIR), args)
    cleaned = matrixgame35_preprocess_text(prompt)
    reference_tokens = reference([cleaned], **pipeline_config.text_encoder_configs[0].tokenizer_kwargs)
    production_tokens = production([cleaned], **pipeline_config.text_encoder_configs[0].tokenizer_kwargs)

    assert torch.equal(production_tokens.input_ids, reference_tokens.input_ids)
    assert torch.equal(
        production_tokens.attention_mask,
        reference_tokens.attention_mask,
    )
    assert reference_tokens.input_ids.shape == reference_tokens.attention_mask.shape == (1, 512)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for UMT5 parity",
)
def test_matrixgame35_wan22_text_encoder_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    input_ids, attention_mask = _official_tokens(
        "A robot walks through a quiet workshop."
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    official = _load_official_text_encoder(device)
    with torch.inference_mode():
        official_hidden = official(input_ids, mask=attention_mask).float().cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_text_encoder(monkeypatch)
    with torch.inference_mode(), set_forward_context(
        current_timestep=0,
        attn_metadata=None,
    ):
        fastvideo_hidden = fastvideo(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state.float().cpu()

    assert official_hidden.shape == fastvideo_hidden.shape == (1, 512, 4096)
    assert_close(fastvideo_hidden, official_hidden, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for UMT5 parity",
)
def test_matrixgame35_diffusers_wan22_text_encoder_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare native FastVideo UMT5 to Transformers from one snapshot."""
    device = torch.device("cuda:0")
    tokenizer = _diffusers_tokenizer()
    config = make_matrixgame35_text_encoder_config()
    tokens = tokenizer(
        [matrixgame35_preprocess_text("A robot walks through a quiet workshop.")],
        **config.tokenizer_kwargs,
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    official = _load_transformers_text_encoder(device)
    with torch.inference_mode():
        official_hidden = official(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state.float().cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_text_encoder(monkeypatch)
    with torch.inference_mode(), set_forward_context(
        current_timestep=0,
        attn_metadata=None,
    ):
        fastvideo_output = fastvideo(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    fastvideo_hidden = fastvideo_output.last_hidden_state.float().cpu()
    valid = attention_mask.bool().cpu()
    official_valid = official_hidden[valid]
    fastvideo_valid = fastvideo_hidden[valid]
    diff = (fastvideo_valid - official_valid).abs()
    print(
        f"transformers UMT5: diff_max={diff.max().item():.6f} "
        f"diff_mean={diff.mean().item():.6f} "
        f"reference_abs_mean={official_valid.abs().mean().item():.6f}"
    )

    assert official_hidden.shape == fastvideo_hidden.shape == (1, 512, 4096)
    assert diff.mean().item() <= 1e-3
    assert_close(fastvideo_valid, official_valid, atol=1e-2, rtol=1e-2)
