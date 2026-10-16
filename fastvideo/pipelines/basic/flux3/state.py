# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from torch import Tensor

from fastvideo.models.dits.flux3 import Packed
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

RUNTIME_STATE_KEY = "_flux3_runtime"


@dataclass
class Flux3RuntimeState:
    packed: Packed | None = None
    image_ids: Tensor | None = None
    text_ids: Tensor | None = None
    crop: tuple[int, int, int, int] | None = None


def get_state(batch: ForwardBatch) -> Flux3RuntimeState:
    state = batch.extra.get(RUNTIME_STATE_KEY)
    if not isinstance(state, Flux3RuntimeState):
        raise RuntimeError("FLUX.3 runtime state has not been initialized")
    return state
