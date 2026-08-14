# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic consistency distillation method."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (SelfForcingFlowMatchScheduler)
from fastvideo.train.callbacks.validation import ValidationCallback
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import CausalModelBase, ModelBase
from fastvideo.train.utils.instantiate import instantiate, resolve_target
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.training.training_utils import EMA_FSDP


def _canonical_parameter_name(name: str) -> str:
    return (name.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.",
                                                                    "").replace("_orig_mod.", ""))


class _EMAState:
    """DCP state for the authoritative CD target EMA shadow."""

    def __init__(self, method: ConsistencyDistillationMethod) -> None:
        self._method = method

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": self._method._target_ema.state_dict(),
            "update_count": torch.tensor(self._method._ema_update_count, dtype=torch.long),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        shadow = state_dict.get("shadow", {})
        if not isinstance(shadow, dict):
            raise TypeError("Consistency EMA checkpoint shadow must be a mapping")
        self._method._target_ema.load_state_dict(shadow)
        update_count = state_dict.get("update_count", 0)
        if torch.is_tensor(update_count):
            update_count = int(update_count.item())
        self._method._ema_update_count = int(update_count)
        self._method._ema_target_dirty = True
        self._method._sync_ema_target()


class ConsistencyDistillationMethod(TrainingMethod):
    """Online consistency distillation for flow-matching models.
    """

    loss_name = "cd_loss"

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        for role in ("student", "teacher"):
            if role not in role_models:
                raise ValueError(f"Consistency distillation requires role {role!r} "
                                 "(student trainable; teacher frozen)")
        if "ema" in role_models:
            raise ValueError("Consistency distillation owns its EMA target; remove models.ema from the config")
        if not self.student._trainable:
            raise ValueError("Consistency distillation requires student to be trainable")
        self.teacher = role_models["teacher"]

        student_model_config = dict(getattr(cfg, "models", {}).get("student", {}))
        if not student_model_config:
            raise ValueError("Consistency distillation requires models.student config to construct its EMA target")
        ema_target_model = instantiate(
            student_model_config,
            training_config=cfg.training,
        )
        if not isinstance(ema_target_model, ModelBase):
            raise TypeError("Consistency distillation EMA target must be a ModelBase, "
                            f"got {type(ema_target_model).__name__}")
        ema_target_model.transformer.requires_grad_(False)
        ema_target_model.transformer.eval()
        ema_target_model._trainable = False
        self._ema_target_model = ema_target_model

        models = (self.student, self.teacher, self._ema_target_model)
        causal_flags = tuple(isinstance(model, CausalModelBase) for model in models)
        if any(causal_flags) and not all(causal_flags):
            raise ValueError("Consistency distillation student, teacher, and internal EMA target must all be "
                             "causal or all be bidirectional")
        self._is_causal = all(causal_flags)

        self._attn_kind = self._infer_attn_kind()
        self._guidance_scale = float(self.method_config.get("guidance_scale", 3.0))
        self._discrete_cd_n = int(self.method_config.get("discrete_cd_N", 48))
        if self._discrete_cd_n < 2:
            raise ValueError("method.discrete_cd_N must be >= 2")
        if "ema_start_step" in self.method_config:
            raise ValueError("method.ema_start_step is not supported by consistency distillation: "
                             "the online target EMA must update after every student optimizer step")
        self._ema_decay = float(self.method_config.get("ema_decay", 0.99))
        if not 0.0 <= self._ema_decay < 1.0:
            raise ValueError("method.ema_decay must be in [0, 1)")
        shift = getattr(self.training_config.pipeline_config, "flow_shift", None)
        self._flow_shift = float(shift) if shift else 5.0

        callbacks = getattr(cfg, "callbacks", {}) or {}
        for callback_name, callback_config in callbacks.items():
            callback_target = str((callback_config or {}).get("_target_", ""))
            if callback_name == "ema":
                raise ValueError("Consistency distillation owns its target EMA; remove callbacks.ema "
                                 "to avoid maintaining a second, divergent EMA")
            if not callback_target:
                if callback_name.startswith("validation"):
                    raise ValueError(
                        "Consistency distillation validation must use "
                        "ConsistencyDistillationValidationCallback so it evaluates the method-owned EMA target")
                continue
            callback_cls = resolve_target(callback_target)
            from fastvideo.train.callbacks.ema import EMACallback
            if isinstance(callback_cls, type) and issubclass(callback_cls, EMACallback):
                raise ValueError("Consistency distillation owns its target EMA; remove callbacks.ema "
                                 "to avoid maintaining a second, divergent EMA")
            if (isinstance(callback_cls, type) and issubclass(callback_cls, ValidationCallback)
                    and not issubclass(callback_cls, ConsistencyDistillationValidationCallback)):
                raise ValueError(
                    "Consistency distillation validation must use "
                    "ConsistencyDistillationValidationCallback so it evaluates the method-owned EMA target")

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
        self._target_ema = EMA_FSDP(
            self.student.transformer,
            decay=self._ema_decay,
            mode="local_shard",
        )
        self._ema_update_count = 0
        self._ema_target_dirty = True
        self._sync_ema_target()
        self._init_optimizers_and_schedulers()

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

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["consistency_distillation.ema"] = _EMAState(self)
        return states

    def get_ema_target_model(self) -> ModelBase:
        """Return the synchronized CD-owned model used for evaluation/export."""
        self._sync_ema_target()
        return self._ema_target_model

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
            raise ValueError("Consistency distillation expects [B, T, C, H, W] latents")

        batch_size, num_latents = int(clean_latents.shape[0]), int(clean_latents.shape[1])
        device = clean_latents.device

        sigmas = self._sf_scheduler.sigmas.to(device)
        timesteps = self._sf_scheduler.timesteps.to(device)
        idx = torch.randint(0, self._discrete_cd_n - 1, (1, ), generator=self.cuda_generator, device=device).squeeze(0)
        t, t_next = timesteps[idx], timesteps[idx + 1]
        sigma_t, sigma_t_next = sigmas[idx], sigmas[idx + 1]
        model_t = self._model_timestep(t, batch_size=batch_size, num_latents=num_latents, device=device)
        model_t_next = self._model_timestep(t_next, batch_size=batch_size, num_latents=num_latents, device=device)

        noise = torch.randn(
            clean_latents.shape,
            generator=self.cuda_generator,
            device=device,
            dtype=clean_latents.dtype,
        )
        latent_t = (1.0 - sigma_t) * clean_latents + sigma_t * noise

        # predict_noise also feeds batch.timesteps into forward context. Keep
        # it synchronized with the explicit timestep for every model call.
        training_batch.timesteps = model_t
        with torch.no_grad():
            v_cond = self._predict_flow(self.teacher,
                                        latent_t,
                                        model_t,
                                        training_batch,
                                        conditional=True,
                                        clean_latents=clean_latents)
            v_uncond = self._predict_flow(self.teacher,
                                          latent_t,
                                          model_t,
                                          training_batch,
                                          conditional=False,
                                          clean_latents=clean_latents)
            v_pred = v_uncond + self._guidance_scale * (v_cond - v_uncond)
            dt = (t - t_next) / float(self.student.num_train_timesteps)
            latent_t_next = latent_t - dt * v_pred

        flow_student = self._predict_flow(self.student,
                                          latent_t,
                                          model_t,
                                          training_batch,
                                          conditional=True,
                                          clean_latents=clean_latents)
        x0_t = latent_t - sigma_t * flow_student

        training_batch.timesteps = model_t_next
        with torch.no_grad():
            # The CPU float32 shadow is authoritative. Synchronize it into the
            # separately callable frozen target immediately before use, as in
            # the Causal-Forcing reference implementation.
            self._sync_ema_target()
            flow_ema = self._predict_flow(self._ema_target_model,
                                          latent_t_next,
                                          model_t_next,
                                          training_batch,
                                          conditional=True,
                                          clean_latents=clean_latents)
            x0_t_next = latent_t_next - sigma_t_next * flow_ema

        loss = F.mse_loss(x0_t.float(), x0_t_next.float())
        loss_map = {"total_loss": loss, self.loss_name: loss}
        attn_metadata = (training_batch.attn_metadata_vsa if self._attn_kind == "vsa" else training_batch.attn_metadata)
        outputs: dict[str, Any] = {"_fv_backward": (model_t, attn_metadata)}
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

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
        # The EMA model is the online consistency target, so it must follow
        # every completed student optimizer step. Delaying these updates
        # leaves the target anchored to initialization and changes the CD
        # objective during early training.
        self._update_ema()

    def _model_timestep(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        num_latents: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self._is_causal:
            return timestep * torch.ones(batch_size, num_latents, device=device)
        return timestep * torch.ones(batch_size, device=device)

    def _predict_flow(
        self,
        model: ModelBase,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_causal:
            return model.predict_noise(latents,
                                       timestep,
                                       batch,
                                       conditional=conditional,
                                       cfg_uncond=None,
                                       attn_kind=self._attn_kind,
                                       clean_x=clean_latents)  # type: ignore[call-arg]
        return model.predict_noise(latents,
                                   timestep,
                                   batch,
                                   conditional=conditional,
                                   cfg_uncond=None,
                                   attn_kind=self._attn_kind)

    @torch.no_grad()
    def _update_ema(self) -> None:
        self._target_ema.update(self.student.transformer)
        self._ema_update_count += 1
        self._ema_target_dirty = True

    @torch.no_grad()
    def _sync_ema_target(self) -> None:
        if not self._ema_target_dirty:
            return
        self._target_ema.copy_to_model(
            self._ema_target_model.transformer,
            name_mapper=_canonical_parameter_name,
            strict=True,
        )
        self._ema_target_dirty = False

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.optimizer.learning_rate must be > 0 for consistency distillation")
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


class ConsistencyDistillationValidationCallback(ValidationCallback):
    """Validate the EMA target owned by ``ConsistencyDistillationMethod``."""

    def _run_validation(
        self,
        method: TrainingMethod,
        step: int,
    ) -> None:
        if not isinstance(method, ConsistencyDistillationMethod):
            raise TypeError("ConsistencyDistillationValidationCallback requires "
                            "ConsistencyDistillationMethod")
        transformer = method.get_ema_target_model().transformer
        with self._validation_memory_context(
                method,
                validation_transformer=transformer,
        ), self._attn_qat_infer_context(transformer):
            self._run_validation_inner(
                method,
                step,
                transformer,
            )
