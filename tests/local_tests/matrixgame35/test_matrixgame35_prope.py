# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from fastvideo.models.dits._matrixgame35_prope import (
    apply_matrixgame35_prope_qkv,
    prope_dot_product_attention,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_SOURCE = _REPO_ROOT / "Matrix-Game-3.5" / "diffsynth" / "models" / "prope_attention.py"


def _load_upstream_prope() -> ModuleType:
    if not _UPSTREAM_SOURCE.is_file():
        pytest.skip(f"Pinned Matrix-Game 3.5 source not found at {_UPSTREAM_SOURCE}.")
    spec = importlib.util.spec_from_file_location("matrixgame35_upstream_prope", _UPSTREAM_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream PRoPE source at {_UPSTREAM_SOURCE}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_prope() -> ModuleType:
    return _load_upstream_prope()


@pytest.fixture
def parity_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    generator = torch.Generator().manual_seed(314159)
    batch, heads, camera_frames, head_dim = 2, 3, 3, 128
    query = torch.randn(batch, heads, 6, head_dim, generator=generator, dtype=torch.float64)
    key = torch.randn(batch, heads, 9, head_dim, generator=generator, dtype=torch.float64)
    value = torch.randn(batch, heads, 9, head_dim, generator=generator, dtype=torch.float64)

    projection = torch.eye(4, dtype=torch.float64).repeat(batch, camera_frames, 4, 1, 1)
    projection[..., :3, :3] += 0.05 * torch.randn(
        batch, camera_frames, 4, 3, 3, generator=generator, dtype=torch.float64
    )
    projection[..., :3, 3] = 0.1 * torch.randn(
        batch, camera_frames, 4, 3, generator=generator, dtype=torch.float64
    )
    transpose = projection.transpose(-1, -2)
    inverse = torch.linalg.inv(projection)
    return query, key, value, (projection, transpose, inverse)


def test_full_layout_qkv_transforms_match_upstream(upstream_prope, parity_inputs) -> None:
    query, key, value, viewmats = parity_inputs
    apply_query, apply_key_value, _ = upstream_prope._prepare_apply_fns(
        head_dim=query.shape[-1],
        viewmats=viewmats,
        camera_layout="full",
    )

    expected = (apply_query(query), apply_key_value(key), apply_key_value(value))
    actual = apply_matrixgame35_prope_qkv(query, key, value, viewmats=viewmats)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


def test_full_layout_sdpa_output_matches_upstream(upstream_prope, parity_inputs) -> None:
    query, key, value, viewmats = parity_inputs
    attention_mask = torch.zeros(query.shape[2], key.shape[2], dtype=query.dtype)
    attention_mask[:, -1] = -0.75

    expected = upstream_prope.prope_dot_product_attention(
        query,
        key,
        value,
        viewmats=viewmats,
        camera_layout="full",
        scale=0.125,
        attn_mask=attention_mask,
        dropout_p=0.0,
    )
    actual = prope_dot_product_attention(
        query,
        key,
        value,
        viewmats=viewmats,
        scale=0.125,
        attn_mask=attention_mask,
        dropout_p=0.0,
    )

    assert actual.shape == query.shape
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_rejects_unreleased_camera_layout(parity_inputs) -> None:
    query, key, value, viewmats = parity_inputs
    with pytest.raises(ValueError, match="camera_layout='full'"):
        apply_matrixgame35_prope_qkv(
            query,
            key,
            value,
            viewmats=viewmats,
            camera_layout="sf13",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda query, key, value, viewmats: (query[..., :96], key[..., :96], value[..., :96], viewmats),
         "head_dim must be a positive multiple of 64"),
        (lambda query, key, value, viewmats: (query[:, :, :5], key, value, viewmats),
         "query sequence length 5 must be divisible"),
        (lambda query, key, value, viewmats: (query, key[..., :-1], value, viewmats),
         "key and value shapes must match"),
        (lambda query, key, value, viewmats: (query, key, value, viewmats[:2]),
         "viewmats must contain exactly"),
    ),
)
def test_rejects_invalid_tensor_contract(parity_inputs, mutation, message) -> None:
    query, key, value, viewmats = mutation(*parity_inputs)
    with pytest.raises(ValueError, match=message):
        apply_matrixgame35_prope_qkv(query, key, value, viewmats=viewmats)
