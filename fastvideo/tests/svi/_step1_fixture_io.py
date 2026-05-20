# SPDX-License-Identifier: Apache-2.0
"""Shared load/save helpers for the Stage 6 single-step parity fixture.

The fixture lives at ``outputs/parity_fixture/step01/`` and contains two files:

  - ``inputs.pt`` — produced by the upstream dump driver, consumed by BOTH
    upstream and FV. Holds the deterministic batch + replacement RNG draws
    + initial LoRA state.
  - ``upstream_intermediates.pt`` — produced by upstream's wrapped methods,
    consumed only by the comparator.

See ``/home/hal-kaiqin/.claude/plans/i-want-to-port-snappy-haven.md`` for the
field-by-field schema and tolerance rationale.
"""

from __future__ import annotations

import os
from typing import Any

import torch

INPUTS_FILENAME = "inputs.pt"
UPSTREAM_FILENAME = "upstream_intermediates.pt"

# ---- expected schemas (documented; not strictly enforced at load time) -------

INPUT_KEYS: tuple[str, ...] = (
    "text",                # str
    "video",               # [1, 3, T, H, W] fp32 in [-1, 1]
    "first_ref_frames",    # list of 12 uint8 tensors [H, W, 3]
    "random_ref_frame",    # uint8 tensor [H, W, 3]
    "noise",               # [1, 16, T_lat, H_lat, W_lat] fp32
    "timestep_id",         # int64 scalar (or [1] tensor)
    "lora_init_state",     # dict[str, Tensor]
    "seed",                # int
)

UPSTREAM_KEYS: tuple[str, ...] = (
    "prompt_emb_context",  # [1, 1, 512, 4096] bf16
    "clean_latents",       # [1, 16, T_lat, H_lat, W_lat] bf16
    "clip_feature",        # [1, 257, 1280] bf16
    "y",                   # [1, 20, T_lat, H_lat, W_lat] bf16
    "noisy_latents",       # bf16
    "target",              # bf16
    "pred",                # bf16
    "loss_unweighted",     # fp32 scalar
    "training_weight",     # fp32 scalar
    "loss",                # fp32 scalar
    "timestep_actual",     # fp32 scalar
)


def fixture_dir(root: str = "outputs/parity_fixture/step01") -> str:
    return os.path.abspath(root)


def inputs_path(root: str = "outputs/parity_fixture/step01") -> str:
    return os.path.join(fixture_dir(root), INPUTS_FILENAME)


def upstream_path(root: str = "outputs/parity_fixture/step01") -> str:
    return os.path.join(fixture_dir(root), UPSTREAM_FILENAME)


def save_inputs(payload: dict[str, Any], root: str = "outputs/parity_fixture/step01") -> str:
    os.makedirs(fixture_dir(root), exist_ok=True)
    path = inputs_path(root)
    _ensure_cpu(payload)
    torch.save(payload, path)
    return path


def load_inputs(root: str = "outputs/parity_fixture/step01") -> dict[str, Any]:
    return torch.load(inputs_path(root), map_location="cpu", weights_only=False)


def save_upstream(payload: dict[str, Any], root: str = "outputs/parity_fixture/step01") -> str:
    os.makedirs(fixture_dir(root), exist_ok=True)
    path = upstream_path(root)
    _ensure_cpu(payload)
    torch.save(payload, path)
    return path


def load_upstream(root: str = "outputs/parity_fixture/step01") -> dict[str, Any]:
    return torch.load(upstream_path(root), map_location="cpu", weights_only=False)


def _ensure_cpu(d: dict[str, Any]) -> None:
    """Coerce tensor values to CPU + detached in place (recurses into lists)."""
    for k, v in list(d.items()):
        if isinstance(v, torch.Tensor):
            d[k] = v.detach().to("cpu").contiguous()
        elif isinstance(v, list):
            d[k] = [
                t.detach().to("cpu").contiguous() if isinstance(t, torch.Tensor) else t
                for t in v
            ]
        elif isinstance(v, dict):
            _ensure_cpu(v)
