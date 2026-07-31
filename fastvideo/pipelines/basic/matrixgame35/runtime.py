# SPDX-License-Identifier: Apache-2.0
"""Small device/offload boundaries shared by Matrix-Game 3.5 stages."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from collections.abc import Callable

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.utils import PRECISION_TO_TYPE


def matrixgame35_autocast_context(
    device: torch.device,
    dtype: torch.dtype,
    disabled: bool,
):
    """Use CUDA autocast only when the requested component precision needs it."""
    if device.type != "cuda" or dtype == torch.float32 or disabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _module_dtype(module: Any, fallback: torch.dtype) -> torch.dtype:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        parameter = next(parameters(), None)
        if parameter is not None and parameter.dtype.is_floating_point:
            return parameter.dtype
    return fallback


def run_matrixgame35_vae_operation(
    vae: Any,
    value: torch.Tensor,
    *,
    precision: str,
    device: torch.device,
    fastvideo_args: FastVideoArgs,
    operation: Callable[[Any, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run one VAE operation on the execution device and honor CPU offload."""
    requested_dtype = PRECISION_TO_TYPE[precision]
    vae.to(device)
    autocast_enabled = (device.type == "cuda" and requested_dtype != torch.float32
                        and not fastvideo_args.disable_autocast)
    input_dtype = requested_dtype if autocast_enabled else _module_dtype(vae, requested_dtype)
    value = value.to(device=device, dtype=input_dtype)
    try:
        with matrixgame35_autocast_context(
                device,
                requested_dtype,
                fastvideo_args.disable_autocast,
        ):
            return operation(vae, value)
    finally:
        if getattr(fastvideo_args, "vae_cpu_offload", False):
            vae.to("cpu")


def move_matrixgame35_transformer_for_forward(
    transformer: Any,
    *,
    device: torch.device,
    fastvideo_args: FastVideoArgs,
) -> None:
    """Move a whole-model DiT unless FSDP or layerwise offload owns placement."""
    if not getattr(fastvideo_args, "dit_layerwise_offload", False) and not getattr(
            fastvideo_args,
            "use_fsdp_inference",
            False,
    ):
        transformer.to(device)


def offload_matrixgame35_transformer(
    transformer: Any,
    *,
    fastvideo_args: FastVideoArgs,
) -> None:
    """Park a whole-model DiT after use when requested by the runtime."""
    if (getattr(fastvideo_args, "dit_cpu_offload", False)
            and not getattr(fastvideo_args, "dit_layerwise_offload", False)
            and not getattr(fastvideo_args, "use_fsdp_inference", False)):
        transformer.to("cpu")


__all__ = [
    "matrixgame35_autocast_context",
    "move_matrixgame35_transformer_for_forward",
    "offload_matrixgame35_transformer",
    "run_matrixgame35_vae_operation",
]
