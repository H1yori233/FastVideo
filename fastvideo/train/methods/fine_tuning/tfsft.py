# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing SFT method (TFSFT; algorithm layer)."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.fine_tuning.dfsft import (
    DiffusionForcingSFTMethod, )
from fastvideo.train.models.base import CausalModelBase
from fastvideo.logger import init_logger


logger = init_logger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class TeacherForcingSFTMethod(DiffusionForcingSFTMethod):

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        self._streaming_teacher_forcing = _as_bool(
            self.method_config.get("streaming_teacher_forcing", False))
        self._logged_streaming_teacher_forcing = False

    def _predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        training_batch: Any,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self._streaming_teacher_forcing:
            return self._predict_noise_streaming_teacher_forcing(
                noisy_latents,
                timestep,
                training_batch,
                clean_latents,
            )
        return self.student.predict_noise(
            noisy_latents,
            timestep,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
            clean_x=clean_latents,
        )

    def _predict_noise_streaming_teacher_forcing(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        training_batch: Any,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(self.student, CausalModelBase):
            raise ValueError("streaming_teacher_forcing requires a causal "
                             "student implementing CausalModelBase")

        chunk = int(self._chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")
        if not self._logged_streaming_teacher_forcing:
            logger.info(
                "Using streaming teacher-forcing path "
                "(chunk_size=%d, attn_kind=%s)",
                chunk,
                self._attn_kind,
            )
            self._logged_streaming_teacher_forcing = True

        batch_size, num_latents = noisy_latents.shape[:2]
        cache_tag = "teacher_forcing"
        self.student.clear_caches(cache_tag=cache_tag)

        preds: list[torch.Tensor] = []
        try:
            for start in range(0, int(num_latents), chunk):
                end = min(start + chunk, int(num_latents))
                noisy_block = noisy_latents[:, start:end]
                clean_block = clean_latents[:, start:end]
                timestep_block = timestep[:, start:end]

                pred = self.student.predict_noise_streaming(
                    noisy_block,
                    timestep_block,
                    training_batch,
                    conditional=True,
                    cache_tag=cache_tag,
                    store_kv=False,
                    cur_start_frame=start,
                    attn_kind=self._attn_kind,
                )
                if pred is None:
                    raise RuntimeError("predict_noise_streaming returned "
                                       "None for teacher-forcing prediction")
                preds.append(pred)

                clean_timestep = torch.zeros(
                    (batch_size, end - start),
                    device=clean_latents.device,
                    dtype=timestep.dtype,
                )
                _ = self.student.predict_noise_streaming(
                    clean_block,
                    clean_timestep,
                    training_batch,
                    conditional=True,
                    cache_tag=cache_tag,
                    store_kv=True,
                    cur_start_frame=start,
                    attn_kind=self._attn_kind,
                )
        finally:
            self.student.clear_caches(cache_tag=cache_tag)

        return torch.cat(preds, dim=1)
