# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.methods.consistency_model.consistency_distillation import (
        ConsistencyDistillationMethod,
        ConsistencyDistillationValidationCallback,
    )

__all__ = [
    "ConsistencyDistillationMethod",
    "ConsistencyDistillationValidationCallback",
]


def __getattr__(name: str) -> object:
    if name == "ConsistencyDistillationMethod":
        from fastvideo.train.methods.consistency_model.consistency_distillation import (
            ConsistencyDistillationMethod, )

        return ConsistencyDistillationMethod
    if name == "ConsistencyDistillationValidationCallback":
        from fastvideo.train.methods.consistency_model.consistency_distillation import (
            ConsistencyDistillationValidationCallback, )

        return ConsistencyDistillationValidationCallback
    raise AttributeError(name)
