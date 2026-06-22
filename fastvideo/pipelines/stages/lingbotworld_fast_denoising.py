# SPDX-License-Identifier: Apache-2.0
"""Block-causal DMD denoising for LingBot-World-Fast.

Generates latent frames chunk-by-chunk. Within a chunk, attention is
bidirectional; across chunks it is causal via a rolling KV cache. Each chunk is
denoised with a short DMD timestep schedule, then re-run once at ~zero timestep
so the cache holds the clean (denoised) context the next chunk attends to.
"""
from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)

# DMD timestep indices selected from the 1000-step shifted UniPC schedule.
# Matches the official generate_fast.py default (timesteps_index).
FAST_TIMESTEPS_INDEX = (0, 179, 358, 679)


class LingBotWorldFastCausalDenoisingStage(DenoisingStage):

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(transformer, scheduler, pipeline, transformer_2=None, vae=vae)
        self.num_transformer_blocks = len(self.transformer.blocks)
        self.num_frame_per_block = getattr(self.transformer, "num_frame_per_block", 3)
        self.local_attn_size = getattr(self.transformer, "local_attn_size", -1)
        assert self.num_frame_per_block > 0

    def _select_timesteps(self, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        shift = fastvideo_args.pipeline_config.flow_shift
        num_train = self.scheduler.config.num_train_timesteps
        device = get_local_torch_device()
        try:
            # FastVideo FlowUniPCMultistepScheduler takes shift directly.
            self.scheduler.set_timesteps(num_inference_steps=num_train, device=device, shift=shift)
        except TypeError:
            # diffusers UniPCMultistepScheduler (use_flow_sigmas): flow_shift is config-level.
            if shift is not None and hasattr(self.scheduler.config, "flow_shift"):
                self.scheduler.config.flow_shift = shift
            self.scheduler.set_timesteps(num_inference_steps=num_train, device=device)
        idx = torch.tensor(FAST_TIMESTEPS_INDEX, dtype=torch.long)
        return self.scheduler.timesteps[idx].to(device)

    def _kv_cache_size(self, num_latent_frames: int) -> int:
        if self.local_attn_size != -1:
            return self.local_attn_size * self.frame_seq_length
        return num_latent_frames * self.frame_seq_length

    def _initialize_kv_cache(self, batch_size, num_latent_frames, dtype, device):
        heads = self.transformer.num_attention_heads
        head_dim = getattr(self.transformer, "attention_head_dim", self.transformer.hidden_size // heads)
        size = self._kv_cache_size(num_latent_frames)
        return [{
            "k": torch.zeros([batch_size, size, heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, size, heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype, device):
        heads = self.transformer.num_attention_heads
        head_dim = getattr(self.transformer, "attention_head_dim", self.transformer.hidden_size // heads)
        return [{
            "k": torch.zeros([batch_size, max_text_len, heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, max_text_len, heads, head_dim], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]

    def _camera_chunk(self, batch: ForwardBatch, start_index: int, num_frames: int):
        c2ws = getattr(batch, "c2ws_plucker_emb", None)
        if c2ws is None:
            return None
        return c2ws[:, :, start_index:start_index + num_frames]

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents
        b, c, t, h, w = latents.shape
        self.frame_seq_length = (h * w) // (self.transformer.patch_size[-1] * self.transformer.patch_size[-2])

        if t % self.num_frame_per_block != 0:
            raise ValueError("num_latent_frames must be divisible by num_frame_per_block")

        timesteps = self._select_timesteps(fastvideo_args)
        prompt_embeds = batch.prompt_embeds
        context_noise = int(getattr(fastvideo_args.pipeline_config, "context_noise", 0))

        kv_cache = self._initialize_kv_cache(b, t, target_dtype, latents.device)
        crossattn_cache = self._initialize_crossattn_cache(b, self.transformer.text_len, target_dtype, latents.device)

        image_latent = batch.image_latent

        num_blocks = t // self.num_frame_per_block
        start_index = 0
        with self.progress_bar(total=num_blocks * len(timesteps)) as progress_bar:
            for _ in range(num_blocks):
                nf = self.num_frame_per_block
                current_latents = latents[:, :, start_index:start_index + nf]
                cond_chunk = (image_latent[:, :, start_index:start_index + nf] if image_latent is not None else None)
                cam_chunk = self._camera_chunk(batch, start_index, nf)

                current_latents = self._process_single_block(current_latents, prompt_embeds, cond_chunk, cam_chunk,
                                                             start_index, nf, timesteps, kv_cache, crossattn_cache,
                                                             batch, target_dtype, autocast_enabled, progress_bar)

                latents[:, :, start_index:start_index + nf] = current_latents

                self._update_context_cache(current_latents, prompt_embeds, cond_chunk, cam_chunk, start_index, nf,
                                           context_noise, kv_cache, crossattn_cache, batch, target_dtype,
                                           autocast_enabled)

                start_index += nf

        batch.latents = latents
        return batch

    def _denoise_kwargs(self, cond_chunk, cam_chunk, start_index, kv_cache, crossattn_cache):
        kwargs: dict[str, Any] = {
            "kv_cache": kv_cache,
            "crossattn_cache": crossattn_cache,
            "current_start": start_index * self.frame_seq_length,
            "start_frame": start_index,
            "encoder_hidden_states_image": [],
        }
        if cam_chunk is not None:
            kwargs["c2ws_plucker_emb"] = cam_chunk
        return kwargs

    def _model_input(self, current_latents, cond_chunk, target_dtype):
        x = current_latents.to(target_dtype)
        if cond_chunk is not None:
            x = torch.cat([x, cond_chunk.to(target_dtype)], dim=1)
        return x

    def _process_single_block(self, current_latents, prompt_embeds, cond_chunk, cam_chunk, start_index, nf, timesteps,
                              kv_cache, crossattn_cache, batch, target_dtype, autocast_enabled, progress_bar):
        model_kwargs = self._denoise_kwargs(cond_chunk, cam_chunk, start_index, kv_cache, crossattn_cache)
        for i, t_cur in enumerate(timesteps):
            noise_input = current_latents.permute(0, 2, 1, 3, 4).clone()
            latent_model_input = self._model_input(current_latents, cond_chunk, target_dtype)
            t_expand = t_cur * torch.ones(
                (current_latents.shape[0], nf), device=current_latents.device, dtype=torch.long)
            t_flat = t_cur.repeat(current_latents.shape[0] * nf)

            with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled), \
                 set_forward_context(current_timestep=i, attn_metadata=None, forward_batch=batch):
                pred_noise = self.transformer(latent_model_input, prompt_embeds, t_expand,
                                              **model_kwargs).permute(0, 2, 1, 3, 4)

            pred_video = pred_noise_to_pred_video(pred_noise=pred_noise.flatten(0, 1),
                                                  noise_input_latent=noise_input.flatten(0, 1),
                                                  timestep=t_flat,
                                                  scheduler=self.scheduler).unflatten(0, pred_noise.shape[:2])

            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1] * torch.ones([1], dtype=torch.long, device=pred_video.device)
                noise = torch.randn(
                    pred_video.shape,
                    dtype=pred_video.dtype,
                    generator=(batch.generator[0] if isinstance(batch.generator, list) else batch.generator)).to(
                        pred_video.device)
                renoised = self.scheduler.add_noise(pred_video.flatten(0, 1), noise.flatten(0, 1),
                                                    next_t).unflatten(0, pred_video.shape[:2])
                current_latents = renoised.permute(0, 2, 1, 3, 4)
            else:
                current_latents = pred_video.permute(0, 2, 1, 3, 4)
            if progress_bar is not None:
                progress_bar.update()
        return current_latents

    def _update_context_cache(self, current_latents, prompt_embeds, cond_chunk, cam_chunk, start_index, nf,
                              context_noise, kv_cache, crossattn_cache, batch, target_dtype, autocast_enabled):
        t_context = torch.ones([current_latents.shape[0], nf], device=current_latents.device,
                               dtype=torch.long) * context_noise
        model_kwargs = self._denoise_kwargs(cond_chunk, cam_chunk, start_index, kv_cache, crossattn_cache)
        latent_model_input = self._model_input(current_latents, cond_chunk, target_dtype)
        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled), \
             set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=batch):
            self.transformer(latent_model_input, prompt_embeds, t_context, **model_kwargs)

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_latent", batch.image_latent, V.none_or_tensor_with_dims(5))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
