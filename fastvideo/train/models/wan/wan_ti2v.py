# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 TI2V training model plugin."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader
from fastvideo.train.utils.moduleloader import load_module_from_path

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


class WanTI2VModel(WanModel):
    """Wan 2.2 TI2V model with clean first-frame conditioning."""

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()

        text_len = (training_config.pipeline_config.text_encoder_configs[0].arch_config.text_len)
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=int(text_len),
            parquet_schema=pyarrow_schema_i2v,
        )
        self.start_step = 0

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        batch = super().prepare_batch(
            raw_batch,
            generator=generator,
            latents_source=latents_source,
        )
        first_frame_latent = self._get_first_frame_latent(raw_batch)
        batch.image_latents = first_frame_latent
        batch.first_frame_conditioning = True
        for condition in (batch.conditional_dict, batch.unconditional_dict):
            if condition is not None:
                condition["first_frame_latent"] = first_frame_latent
        return batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        kwargs = super()._build_distill_input_kwargs(
            noise_input,
            timestep,
            text_dict,
            clean_x=clean_x,
            aug_t=aug_t,
        )
        if text_dict is None:
            return kwargs

        first_frame_latent = text_dict.get("first_frame_latent")
        if first_frame_latent is not None:
            kwargs["hidden_states"], kwargs["timestep"] = self._apply_first_frame_latent(
                kwargs["hidden_states"],
                kwargs["timestep"],
                first_frame_latent,
            )
        return kwargs

    def _get_first_frame_latent(self, raw_batch: dict[str, Any]) -> torch.Tensor:
        value = raw_batch.get("first_frame_latent")
        if value is None:
            raise KeyError("Wan 2.2 TI2V batches must contain a 'first_frame_latent' tensor")
        if not torch.is_tensor(value):
            raise TypeError("Wan 2.2 TI2V first_frame_latent must be a torch.Tensor")
        if value.ndim != 5:
            raise ValueError("Wan 2.2 TI2V first_frame_latent must have shape [B, C, T, H, W], "
                             f"got {tuple(value.shape)}")

        if value.shape[2] < 1:
            raise ValueError("Wan 2.2 TI2V first_frame_latent must contain at least one latent frame")
        value = value[:, :, :1]
        return value.to(device=self.device, dtype=self._get_training_dtype())

    def _apply_first_frame_latent(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        first_frame_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first_frame_latent.ndim != 5 or first_frame_latent.shape[2] < 1:
            raise ValueError("Wan 2.2 TI2V first_frame_latent must have shape [B, C, T, H, W] with T >= 1, "
                             f"got {tuple(first_frame_latent.shape)}")
        conditioning_shape = (
            first_frame_latent.shape[0],
            first_frame_latent.shape[1],
            first_frame_latent.shape[3],
            first_frame_latent.shape[4],
        )
        hidden_shape = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.shape[3],
            hidden_states.shape[4],
        )
        if conditioning_shape != hidden_shape:
            raise ValueError("Wan 2.2 TI2V first_frame_latent batch/channel/spatial dimensions must match "
                             f"hidden_states: {conditioning_shape} vs {hidden_shape}")

        hidden_states = torch.cat(
            [
                first_frame_latent[:, :, :1].to(hidden_states.dtype),
                hidden_states[:, :, 1:],
            ],
            dim=2,
        )
        if not bool(getattr(self.training_config.pipeline_config, "expand_timesteps", False)):
            raise ValueError("Wan 2.2 TI2V training requires pipeline.expand_timesteps=true")
        timestep = self._expand_ti2v_timesteps(timestep, hidden_states)
        return hidden_states, timestep

    def _expand_ti2v_timesteps(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.transformer.patch_size
        post_t = num_frames // patch_t
        post_h = height // patch_h
        post_w = width // patch_w

        token_mask = timestep.new_ones((batch_size, post_t, post_h, post_w))
        token_mask[:, 0] = 0
        token_mask = token_mask.flatten(1)

        if timestep.dim() == 1:
            timestep = timestep[:, None].expand(-1, token_mask.shape[1])
        elif timestep.dim() == 2:
            if timestep.shape[1] < post_t:
                raise ValueError("Wan 2.2 TI2V framewise timestep tensor is shorter than the latent sequence: "
                                 f"got {timestep.shape[1]}, expected at least {post_t}")
            timestep = timestep[:, :post_t].repeat_interleave(post_h * post_w, dim=1)
        else:
            raise ValueError(f"Unsupported Wan 2.2 TI2V timestep shape: {tuple(timestep.shape)}")
        return token_mask * timestep
