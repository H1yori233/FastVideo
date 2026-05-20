# SPDX-License-Identifier: Apache-2.0
"""Per-step JSONL logger.

Captures loss values and metrics returned by ``single_train_step`` into a
JSONL file, one row per iteration. Used as the data source for offline
parity checks (FV vs upstream).
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


def _coerce(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().to("cpu", torch.float32).item())
        return value.detach().to("cpu", torch.float32).flatten().tolist()
    return value


class JSONLLogCallback(Callback):
    """Writes a JSONL line per training step on rank 0.

    Schema: ``{"step": int, "loss/<name>": float, "metric/<name>": value, ...}``.
    """

    def __init__(self, *, output_path: str | None = None, flush_every: int = 1) -> None:
        super().__init__()
        self.output_path = output_path
        self.flush_every = max(1, int(flush_every))
        self._fp: Any = None
        self._buffered: int = 0

    def _resolve_path(self) -> str:
        if self.output_path is not None:
            return self.output_path
        base = str(self.training_config.checkpoint.output_dir) or os.getcwd()
        return os.path.join(base, "train_log.jsonl")

    def on_train_start(self, method: TrainingMethod, iteration: int = 0) -> None:
        if get_world_group().rank != 0:
            return
        path = self._resolve_path()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fp = open(path, "a", encoding="utf-8")
        logger.info("JSONLLogCallback: writing to %s", path)

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        del method
        if self._fp is None:
            return
        row: dict[str, Any] = {"step": int(iteration)}
        for k, v in loss_dict.items():
            row[k] = _coerce(v)
        self._fp.write(json.dumps(row) + "\n")
        self._buffered += 1
        if self._buffered >= self.flush_every:
            self._fp.flush()
            self._buffered = 0

    def on_train_end(self, method: TrainingMethod, iteration: int = 0) -> None:
        if self._fp is None:
            return
        try:
            self._fp.flush()
            self._fp.close()
        finally:
            self._fp = None
