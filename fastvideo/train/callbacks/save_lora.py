# SPDX-License-Identifier: Apache-2.0
"""LoRA-only safetensors checkpoint callback.

Writes a single ``.safetensors`` containing just the trainable LoRA tensors
(``lora_A`` / ``lora_B`` weights) at user-configured cadence and at training
end. The output file uses FV-internal layer naming with a ``.weight``
suffix so it can be consumed by ``LoRAPipeline.set_lora_adapter`` without
further conversion.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)

_GRAD_CKPT_WRAP = "_checkpoint_wrapped_module."


def _strip_wrapper_prefixes(name: str) -> str:
    """Strip activation-checkpointing module-wrapper segments from a key name."""
    while _GRAD_CKPT_WRAP in name:
        name = name.replace(_GRAD_CKPT_WRAP, "")
    return name


def _collect_lora_state_dict(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        clean = _strip_wrapper_prefixes(name)
        tensor = param.detach()
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()  # type: ignore[assignment]
        key = clean if clean.endswith(".weight") else clean + ".weight"
        out[key] = tensor.to("cpu", dtype=torch.bfloat16)
    return out


class SaveLoRACallback(Callback):
    """Periodically dump LoRA-only weights to safetensors.

    Saves to ``<output_dir>/lora/step_<iter>.safetensors`` every
    ``every_steps`` iterations (and once at train end). Set
    ``every_steps=0`` to only save at the end.
    """

    def __init__(
        self,
        *,
        every_steps: int = 0,
        output_subdir: str = "lora",
        also_save_last_as: str | None = "last.safetensors",
    ) -> None:
        super().__init__()
        self.every_steps = int(every_steps)
        self.output_subdir = str(output_subdir)
        self.also_save_last_as = also_save_last_as

    def _save_dir(self) -> str:
        base = str(self.training_config.checkpoint.output_dir) or os.getcwd()
        path = os.path.join(base, self.output_subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def _maybe_save(self, method: TrainingMethod, iteration: int) -> None:
        if get_world_group().rank != 0:
            return
        transformer = method.student.transformer
        sd = _collect_lora_state_dict(transformer)
        if not sd:
            logger.warning("SaveLoRACallback: no LoRA parameters found on transformer; skipping")
            return
        out_dir = self._save_dir()
        path = os.path.join(out_dir, f"step_{iteration:06d}.safetensors")
        from safetensors.torch import save_file
        save_file(sd, path)
        logger.info("SaveLoRACallback: wrote %d LoRA tensors to %s", len(sd), path)
        if self.also_save_last_as:
            last = os.path.join(out_dir, str(self.also_save_last_as))
            save_file(sd, last)
            logger.info("SaveLoRACallback: mirrored last checkpoint to %s", last)

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        if self.every_steps <= 0:
            return
        if iteration % self.every_steps != 0:
            return
        self._maybe_save(method, iteration)

    def on_train_end(self, method: TrainingMethod, iteration: int = 0) -> None:
        self._maybe_save(method, iteration)
