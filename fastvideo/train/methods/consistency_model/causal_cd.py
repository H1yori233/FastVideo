# SPDX-License-Identifier: Apache-2.0
"""Causal consistency distillation method (algorithm layer)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (SelfForcingFlowMatchScheduler)
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import CausalModelBase, ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


logger = init_logger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class CausalConsistencyDistillationMethod(TrainingMethod):

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        for role in ("student", "teacher", "ema"):
            if role not in role_models:
                raise ValueError(f"Causal-CD requires role {role!r} "
                                 "(student trainable; teacher + ema frozen, "
                                 "both initialized from the student's "
                                 "checkpoint)")
        if not self.student._trainable:
            raise ValueError("Causal-CD requires student to be trainable")
        self.teacher = role_models["teacher"]
        self.ema_model = role_models["ema"]

        self._attn_kind = self._infer_attn_kind()
        self._guidance_scale = float(self.method_config.get("guidance_scale", 3.0))
        self._discrete_cd_n = int(self.method_config.get("discrete_cd_N", 48))
        if self._discrete_cd_n < 2:
            raise ValueError("method.discrete_cd_N must be >= 2")
        self._chunk_size = int(self.method_config.get("chunk_size", 3))
        if self._chunk_size <= 0:
            raise ValueError("method.chunk_size must be positive")
        self._streaming_causal_cd = _as_bool(
            self.method_config.get("streaming_causal_cd", False))
        self._logged_streaming_causal_cd = False
        self._ema_decay = float(self.method_config.get("ema_decay", 0.99))
        self._ema_start_step = int(self.method_config.get("ema_start_step", 200))
        shift = getattr(self.training_config.pipeline_config, "flow_shift", None)
        self._flow_shift = float(shift) if shift else 5.0

        self.student.init_preprocessors(self.training_config)
        self._sf_scheduler = SelfForcingFlowMatchScheduler(
            num_inference_steps=self._discrete_cd_n,
            num_train_timesteps=int(self.student.num_train_timesteps),
            shift=self._flow_shift,
            sigma_min=0.0,
            sigma_max=1.0,
            extra_one_step=True,
            training=False,
        )
        self._init_optimizers_and_schedulers()

    # ------------------------------------------------------------------

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    # ------------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        clean_latents = training_batch.latents
        if not torch.is_tensor(clean_latents) or clean_latents.ndim != 5:
            raise ValueError("Causal-CD expects [B, T, C, H, W] latents")
        self._validate_streaming_chunk_size()

        batch_size, num_latents = int(clean_latents.shape[0]), int(clean_latents.shape[1])
        device = clean_latents.device

        sigmas = self._sf_scheduler.sigmas.to(device)
        timesteps = self._sf_scheduler.timesteps.to(device)
        idx = torch.randint(0, self._discrete_cd_n - 1, (1, ), generator=self.cuda_generator, device=device).squeeze(0)
        t, t_next = timesteps[idx], timesteps[idx + 1]
        sigma_t, sigma_t_next = sigmas[idx], sigmas[idx + 1]
        t_pf = t * torch.ones(batch_size, num_latents, device=device)
        t_next_pf = t_next * torch.ones(batch_size, num_latents, device=device)

        noise = torch.randn(
            clean_latents.shape,
            generator=self.cuda_generator,
            device=device,
            dtype=clean_latents.dtype,
        )
        latent_t = (1.0 - sigma_t) * clean_latents + sigma_t * noise

        with torch.no_grad():
            v_cond = self._predict_flow(self.teacher,
                                        latent_t,
                                        t_pf,
                                        training_batch,
                                        conditional=True,
                                        clean_x=clean_latents)
            v_uncond = self._predict_flow(self.teacher,
                                          latent_t,
                                          t_pf,
                                          training_batch,
                                          conditional=False,
                                          clean_x=clean_latents)
            v_pred = v_uncond + self._guidance_scale * (v_cond - v_uncond)
            dt = ((t - t_next) / float(self.student.num_train_timesteps))
            latent_t_next = latent_t - dt * v_pred

        training_batch.timesteps = t_pf
        flow_student = self._predict_flow(self.student,
                                          latent_t,
                                          t_pf,
                                          training_batch,
                                          conditional=True,
                                          clean_x=clean_latents)
        x0_t = latent_t - sigma_t * flow_student

        with torch.no_grad():
            flow_ema = self._predict_flow(self.ema_model,
                                          latent_t_next,
                                          t_next_pf,
                                          training_batch,
                                          conditional=True,
                                          clean_x=clean_latents)
            x0_t_next = latent_t_next - sigma_t_next * flow_ema

        loss = F.mse_loss(x0_t.float(), x0_t_next.float())

        loss_map = {"total_loss": loss, "causal_cd_loss": loss}
        attn_metadata = (training_batch.attn_metadata_vsa if self._attn_kind == "vsa" else training_batch.attn_metadata)
        outputs: dict[str, Any] = {"_fv_backward": (t_pf, attn_metadata)}
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

    # ------------------------------------------------------------------

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        ctx = outputs.get("_fv_backward")
        if ctx is None:
            super().backward(loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return
        self.student.backward(loss_map["total_loss"], ctx, grad_accum_rounds=grad_accum_rounds)

    def optimizers_schedulers_step(self, iteration: int) -> None:
        super().optimizers_schedulers_step(iteration)
        if iteration >= self._ema_start_step:
            self._update_ema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_flow(
        self,
        model: ModelBase,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        clean_x: torch.Tensor,
    ) -> torch.Tensor:
        if self._streaming_causal_cd:
            return self._predict_flow_streaming(
                model,
                latents,
                timestep,
                batch,
                conditional=conditional,
                clean_x=clean_x,
            )
        return model.predict_noise(latents,
                                   timestep,
                                   batch,
                                   conditional=conditional,
                                   cfg_uncond=None,
                                   attn_kind=self._attn_kind,
                                   clean_x=clean_x)

    def _predict_flow_streaming(
        self,
        model: ModelBase,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        clean_x: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(model, CausalModelBase):
            raise ValueError("streaming_causal_cd requires causal role models "
                             "implementing CausalModelBase")
        if not self._logged_streaming_causal_cd:
            logger.info(
                "Using streaming causal-CD path "
                "(chunk_size=%d, attn_kind=%s)",
                self._chunk_size,
                self._attn_kind,
            )
            self._logged_streaming_causal_cd = True

        batch_size, num_latents = latents.shape[:2]
        cache_tag = f"causal_cd_{'cond' if conditional else 'uncond'}"
        model.clear_caches(cache_tag=cache_tag)
        preds: list[torch.Tensor] = []
        try:
            for start in range(0, int(num_latents), self._chunk_size):
                end = min(start + self._chunk_size, int(num_latents))
                pred = model.predict_noise_streaming(
                    latents[:, start:end],
                    timestep[:, start:end],
                    batch,
                    conditional=conditional,
                    cache_tag=cache_tag,
                    store_kv=False,
                    cur_start_frame=start,
                    attn_kind=self._attn_kind,
                )
                if pred is None:
                    raise RuntimeError("predict_noise_streaming returned "
                                       "None for causal-CD prediction")
                preds.append(pred)

                clean_timestep = torch.zeros(
                    (int(batch_size), end - start),
                    device=clean_x.device,
                    dtype=timestep.dtype,
                )
                _ = model.predict_noise_streaming(
                    clean_x[:, start:end],
                    clean_timestep,
                    batch,
                    conditional=conditional,
                    cache_tag=cache_tag,
                    store_kv=True,
                    cur_start_frame=start,
                    attn_kind=self._attn_kind,
                )
        finally:
            model.clear_caches(cache_tag=cache_tag)

        return torch.cat(preds, dim=1)

    def _validate_streaming_chunk_size(self) -> None:
        if not self._streaming_causal_cd:
            return
        expected = getattr(
            self.student.transformer,
            "num_frame_per_block",
            None,
        )
        if expected is not None and int(expected) != int(self._chunk_size):
            raise ValueError("streaming_causal_cd chunk_size must match "
                             "transformer.num_frame_per_block "
                             f"(got {self._chunk_size}, expected {expected})")

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self._ema_decay
        for ema_p, p in zip(self.ema_model.transformer.parameters(), self.student.transformer.parameters(),
                            strict=True):
            ema_p.mul_(decay).add_(p.detach().to(ema_p.dtype), alpha=1.0 - decay)

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for causal-cd")
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )
