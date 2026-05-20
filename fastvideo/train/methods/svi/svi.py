# SPDX-License-Identifier: Apache-2.0
"""Stable-Video-Infinity training method (Stage 2: no Error Recycling).

Mirrors the inference-time SVI pipeline's I2V conditioning surface while
running gradients through a LoRA on the Wan transformer. Encodes text,
video and reference frames on-the-fly per step. Error Recycling will be
bolted on in Stage 5.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import PIL.Image
import torch
import torch.nn.functional as F

from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.svi.error_recycling import (
    ErrorRecyclingConfig,
    ErrorRecyclingState,
    compute_step_errors,
    sigma_at,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

logger = init_logger(__name__)


def _compute_bell_weights(num_steps: int) -> torch.Tensor:
    """Linear-timesteps bell-curve weighting, matching diffsynth.

    The shape is exp(-2 * ((i - N/2) / N) ** 2) shifted to start at zero,
    normalized to sum to N — so a uniform sampler over the schedule lands
    on an average weight of 1.0.
    """
    x = torch.arange(num_steps, dtype=torch.float32)
    y = torch.exp(-2 * ((x - num_steps / 2) / num_steps)**2)
    y_shifted = y - y.min()
    return y_shifted * (num_steps / y_shifted.sum())


class SVITrainingMethod(TrainingMethod):
    """LoRA fine-tune for the SVI flavor of Wan I2V."""

    def __init__(self, *, cfg: Any, role_models: dict[str, ModelBase]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("SVITrainingMethod requires role 'student'")
        if not self.student._trainable:
            raise ValueError("SVITrainingMethod requires student trainable (LoRA enabled)")

        self._attn_kind: Literal["dense", "vsa"] = self._infer_attn_kind()

        mc = self.method_config
        self.num_motion_frames = int(mc.get("num_motion_frames", 1))
        self.ref_pad_num = int(mc.get("ref_pad_num", -1))
        self.ref_pad_cfg = bool(mc.get("ref_pad_cfg", False))
        self.p_motion_threshold = float(mc.get("p_motion_threshold", 0.9))
        self.repeat_first_frame = bool(mc.get("repeat_first_frame", True))

        self.student.init_preprocessors(self.training_config)
        self._init_optimizers_and_schedulers()

        # Per-step training weighting. Upstream calls scheduler.set_timesteps(num_inference_steps=1000, training=True)
        # which produces a bell-curve over the full timestep range — we precompute it once.
        num_train = int(self.student.num_train_timesteps)
        self.register_buffer("_training_weights", _compute_bell_weights(num_train), persistent=False)

        # Error Recycling (Stage 5). Disabled by default; enable in YAML with
        # ``method.use_error_recycling: true``.
        self.use_error_recycling = bool(mc.get("use_error_recycling", False))
        if self.use_error_recycling:
            self._er = ErrorRecyclingState(
                ErrorRecyclingConfig(
                    enabled=True,
                    error_buffer_k=int(mc.get("error_buffer_k", 500)),
                    num_grids=int(mc.get("num_grids", 50)),
                    flow_shift=float(self.student.timestep_shift),
                    buffer_warmup_iter=int(mc.get("buffer_warmup_iter", 50)),
                    buffer_replacement_strategy=str(mc.get("buffer_replacement_strategy", "random")),
                    noise_prob=float(mc.get("noise_prob", 0.99)),
                    y_prob=float(mc.get("y_prob", 0.99)),
                    latent_prob=float(mc.get("latent_prob", 0.99)),
                    clean_prob=float(mc.get("clean_prob", 0.1)),
                    clean_buffer_update_prob=float(mc.get("clean_buffer_update_prob", 0.5)),
                    error_modulate_factor=float(mc.get("error_modulate_factor", 0.0)),
                    y_error_num=int(mc.get("y_error_num", 1)),
                ),
                num_train_timesteps=num_train,
            )
            logger.info("Error Recycling enabled (warmup=%d, num_grids=%d, k=%d, strategy=%s)",
                        self._er.cfg.buffer_warmup_iter, self._er.cfg.num_grids, self._er.cfg.error_buffer_k,
                        self._er.cfg.buffer_replacement_strategy)
        else:
            self._er = None

    # ------------------------------------------------------------------
    # TrainingMethod abstract overrides
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

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        lr = float(tc.optimizer.learning_rate)
        if lr <= 0.0:
            raise ValueError("training.optimizer.learning_rate must be > 0")
        params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters on student.transformer — did LoRA fail to attach?")
        (self._student_optimizer, self._student_lr_scheduler) = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=lr,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    # ------------------------------------------------------------------
    # Single train step
    # ------------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        device = self.student.device
        dtype = self.student._get_training_dtype()

        # ----- Decode batch (B=1 expected for SVI) -----
        text: str = batch["text"][0]
        video: torch.Tensor = batch["video"].to(device=device, dtype=torch.float32)  # [B, 3, T, H, W]
        first_ref_frames_per_sample: list[list[torch.Tensor]] = batch["first_ref_frames"]
        random_ref_tensor: torch.Tensor = batch["random_ref_frame"][0]  # [H, W, 3] uint8

        # ----- Text encode (frozen) -----
        encoder_hidden_states, encoder_attention_mask = self._encode_text(text, device=device, dtype=dtype)

        # ----- VAE encode video -> clean latents -----
        clean_latents = self._vae_encode(video).to(dtype=dtype)  # [B, 16, T_lat, H_lat, W_lat]

        # ----- Pick motion frames (probabilistic, matches upstream) -----
        first_refs_pil = [PIL.Image.fromarray(t.cpu().to(torch.uint8).numpy()) for t in first_ref_frames_per_sample[0]]
        random_ref_pil = PIL.Image.fromarray(random_ref_tensor.cpu().to(torch.uint8).numpy())
        condition_frames = self._select_motion_frames(first_refs_pil)

        # ----- Build y conditioning (CLIP feat + mask + VAE-encoded motion + padding) -----
        b, _, num_frames, height, width = video.shape
        clip_feature, y = self._build_y(
            condition_frames=condition_frames,
            random_ref=random_ref_pil,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

        # ----- Sample noise + timestep -----
        gen = self.cuda_generator
        noise = torch.randn(clean_latents.shape, generator=gen, device=device, dtype=clean_latents.dtype)
        timestep_id = torch.randint(0, int(self.student.num_train_timesteps), (1, ), generator=gen, device=device)
        timestep = self.student.noise_scheduler.timesteps.to(device=device)[timestep_id].to(dtype=torch.float32)

        # ----- Error Recycling: inject sampled errors before add_noise -----
        noise_w_error = noise
        latents_w_error = clean_latents
        er = self._er
        use_clean_input = False
        injected = {"noise": False, "y": False, "latent": False}
        if er is not None:
            grid_idx = er.binner(timestep)
            inj_noise, inj_y, inj_latent, use_clean_input = er.roll()
            if inj_noise:
                e = er.sample_noise_error(grid_idx, clean_latents)
                if e is not None:
                    noise_w_error = noise + e.to(noise.dtype)
                    injected["noise"] = True
            if inj_y:
                e = er.sample_y_error(grid_idx, clean_latents)
                if e is not None:
                    # Upstream injects only into y[:, 4:, :y_error_num, ...] (the VAE-encoded slice
                    # of the motion-frame conditioning).
                    n = min(int(er.cfg.y_error_num), e.shape[2], y.shape[2])
                    max_start = max(0, e.shape[2] - n)
                    start = random.randint(0, max_start) if max_start > 0 else 0
                    slab = e[:, :, start:start + n, ...].to(y.dtype)
                    y = y.clone()
                    y[:, 4:, :n, ...] = y[:, 4:, :n, ...] + slab
                    injected["y"] = True
            if inj_latent:
                e = er.sample_latent_error(grid_idx, clean_latents)
                if e is not None:
                    latents_w_error = clean_latents + e.to(clean_latents.dtype)
                    injected["latent"] = True

        # ----- add_noise (uses scheduler's discrete sigma lookup) -----
        bsz, _, t_lat, _, _ = clean_latents.shape
        noisy_clean_2d = self.student.noise_scheduler.add_noise(
            latents_w_error.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous(),  # [B*T_lat, C, H, W]
            noise_w_error.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous(),
            timestep,
        )
        noisy_latents = noisy_clean_2d.unflatten(0, (bsz, t_lat)).permute(0, 2, 1, 3, 4).contiguous()

        # ----- Build transformer input: hidden_states is [B, 16+20, T_lat, H_lat, W_lat] -----
        hidden_states = torch.cat([noisy_latents, y], dim=1).to(dtype=dtype)

        with torch.autocast(device.type, dtype=dtype), set_forward_context(
                current_timestep=timestep,
                attn_metadata=None,
        ):
            pred = self.student.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_image=[clip_feature],
                timestep=timestep,
                return_dict=False,
            )

        # Transformer returns [B, C_out, T_lat, H_lat, W_lat]; align with target shape.
        if isinstance(pred, tuple):
            pred = pred[0]

        # Upstream: training_target = noise_w_error - latents (note: CLEAN latents, not _w_error).
        # The self-correction trains the model to pull a corrupted noisy_latents back to the clean manifold.
        target = (noise_w_error - clean_latents).to(dtype=pred.dtype)
        weight = self._training_weights.to(device=device, dtype=torch.float32)[timestep_id.cpu().item()]
        loss = F.mse_loss(pred.float(), target.float()) * float(weight)

        # ----- Post-loss: push step errors to ER buffers -----
        if er is not None:
            self._update_error_buffers(
                noise_pred=pred,
                training_target=target,
                timestep=timestep,
                iteration=iteration,
                use_clean_input=use_clean_input,
            )

        loss_map = {"total_loss": loss, "svi_loss": loss}
        outputs: dict[str, Any] = {"_fv_backward": (timestep, None)}
        metrics: dict[str, LogScalar] = {
            "train/timestep": float(timestep.item()),
            "train/weight": float(weight),
            "train/er_inject_noise": float(injected["noise"]),
            "train/er_inject_y": float(injected["y"]),
            "train/er_inject_latent": float(injected["latent"]),
            "train/er_use_clean_input": float(use_clean_input),
        }
        if er is not None:
            metrics["train/er_noise_buffer_size"] = float(er.noise_buffer.total_size())
            metrics["train/er_y_buffer_size"] = float(er.y_buffer.total_size())
        return loss_map, outputs, metrics

    # ------------------------------------------------------------------
    # Error Recycling helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_error_buffers(
        self,
        *,
        noise_pred: torch.Tensor,
        training_target: torch.Tensor,
        timestep: torch.Tensor,
        iteration: int,
        use_clean_input: bool,
    ) -> None:
        er = self._er
        assert er is not None  # caller-checked
        # Gate buffer updates by clean_buffer_update_prob when clean input was used.
        if use_clean_input and random.random() >= er.cfg.clean_buffer_update_prob:
            return
        sigma = sigma_at(self.student.noise_scheduler, timestep)
        noise_error, y_error = compute_step_errors(
            noise_pred=noise_pred,
            training_target=training_target,
            sigma=sigma,
        )
        # Keep the [B=1, C, T, H, W] storage so the y-channel slice math at injection
        # time matches upstream verbatim (image_emb["y"][:, 4:, :y_error_num]).
        if er.in_warmup(iteration):
            er.update_distributed(noise_error=noise_error, y_error=y_error, timestep=timestep)
        else:
            er.update_local(noise_error=noise_error, y_error=y_error, timestep=timestep)

    # ------------------------------------------------------------------
    # Backward override: re-enter forward context for grad-ckpt recompute
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
        timesteps, attn_metadata = ctx
        with set_forward_context(current_timestep=timesteps, attn_metadata=attn_metadata):
            (loss_map["total_loss"] / grad_accum_rounds).backward()

    # ------------------------------------------------------------------
    # Encoder helpers (no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text(self, text: str, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        student = self.student
        pipeline_config = self.training_config.pipeline_config
        encoder_config = pipeline_config.text_encoder_configs[0]
        postprocess = pipeline_config.postprocess_text_funcs[0]
        preprocess_funcs = getattr(pipeline_config, "preprocess_text_funcs", None)
        if preprocess_funcs is not None and preprocess_funcs[0] is not None:
            text = preprocess_funcs[0](text)

        tok_kwargs = dict(encoder_config.tokenizer_kwargs)
        student.text_encoder = student.text_encoder.to(device).eval()
        text_inputs = student.tokenizer(text, **tok_kwargs).to(device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = student.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
            outputs.attention_mask = text_inputs["attention_mask"]
            embeds = postprocess(outputs).to(device=device, dtype=dtype)
        mask = text_inputs["attention_mask"].to(device=device, dtype=dtype)
        return embeds, mask

    @torch.no_grad()
    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        """VAE encode a normalized video [B, 3, T, H, W] in [-1, 1] -> latent."""
        student = self.student
        vae = student.vae.to(video.device, dtype=torch.float32).eval()
        with set_forward_context(current_timestep=0, attn_metadata=None):
            enc = vae.encode(video)
        latent = self._retrieve_latent(enc)
        return self._apply_vae_scale(latent, vae)

    @torch.no_grad()
    def _vae_encode_condition_video(self, condition_video: torch.Tensor) -> torch.Tensor:
        return self._vae_encode(condition_video)

    @torch.no_grad()
    def _clip_encode(self, image: PIL.Image.Image, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        student = self.student
        student.image_encoder = student.image_encoder.to(device).eval()
        inputs = student.image_processor(images=image, return_tensors="pt").to(device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = student.image_encoder(**inputs)
        return outputs.last_hidden_state.to(dtype=dtype)

    @staticmethod
    def _retrieve_latent(encoder_output: Any) -> torch.Tensor:
        if hasattr(encoder_output, "latent_dist"):
            return encoder_output.latent_dist.sample()
        if hasattr(encoder_output, "sample") and callable(encoder_output.sample):
            return encoder_output.sample()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        if isinstance(encoder_output, torch.Tensor):
            return encoder_output
        raise TypeError(f"Unrecognised VAE encode output: {type(encoder_output).__name__}")

    @staticmethod
    def _apply_vae_scale(latent: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
        shift = getattr(vae, "shift_factor", None)
        if shift is not None:
            if isinstance(shift, torch.Tensor):
                shift = shift.to(latent.device, latent.dtype)
            latent = latent - shift
        scaling = getattr(vae, "scaling_factor", 1.0)
        if isinstance(scaling, torch.Tensor):
            scaling = scaling.to(latent.device, latent.dtype)
        return latent * scaling

    # ------------------------------------------------------------------
    # y-conditioning (mirror SVIImageVAEEncodingStage)
    # ------------------------------------------------------------------

    def _select_motion_frames(self, first_refs: list[PIL.Image.Image]) -> list[PIL.Image.Image]:
        if self.num_motion_frames <= 1:
            return first_refs[:1]
        if random.random() < self.p_motion_threshold:
            return first_refs[:self.num_motion_frames]
        if self.repeat_first_frame:
            return [first_refs[0]] * self.num_motion_frames
        return first_refs[:1]

    @torch.no_grad()
    def _build_y(
        self,
        *,
        condition_frames: list[PIL.Image.Image],
        random_ref: PIL.Image.Image,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (clip_feature, y) where y has shape [B, 20, T_lat, H_lat, W_lat]."""
        student = self.student
        vae = student.vae
        vae_scale = vae.spatial_compression_ratio
        temporal_compression = vae.temporal_compression_ratio
        latent_h = height // vae_scale
        latent_w = width // vae_scale

        # 1) CLIP feature off the first condition frame (matches inference).
        clip_feature = self._clip_encode(condition_frames[0], device=device, dtype=dtype)

        # 2) Mask: 1 on the first len(condition) frames, 0 elsewhere — then folded to (4, T_lat, ...).
        num_cond = len(condition_frames)
        msk = torch.ones(1, num_frames, latent_h, latent_w, device=device, dtype=torch.float32)
        if self.ref_pad_cfg:
            msk[:, num_cond:] = 0
        else:
            msk[:, 1:] = 0
        msk = torch.cat([msk[:, 0:1].repeat_interleave(temporal_compression, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // temporal_compression, temporal_compression, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]  # (4, T_lat, H_lat, W_lat)

        # 3) Build the VAE-input video: concat(condition_frames, padding).
        cond_video = self._pil_list_to_tensor(condition_frames, height=height, width=width, device=device)
        remaining = num_frames - num_cond
        if remaining < 0:
            raise ValueError(f"num_frames={num_frames} < num_motion={num_cond}")
        if remaining == 0:
            pad_video = cond_video.new_empty(1, 3, 0, height, width)
        elif self.ref_pad_num == 0:
            pad_video = cond_video.new_zeros(1, 3, remaining, height, width)
        elif self.ref_pad_num == -1:
            ref_t = self._pil_list_to_tensor([random_ref], height=height, width=width, device=device)
            pad_video = ref_t.repeat(1, 1, remaining, 1, 1)
        elif self.ref_pad_num > 0:
            ref_t = self._pil_list_to_tensor([random_ref], height=height, width=width, device=device)
            k = min(self.ref_pad_num, remaining)
            head = ref_t.repeat(1, 1, k, 1, 1)
            if remaining > k:
                pad_video = torch.cat([head, ref_t.new_zeros(1, 3, remaining - k, height, width)], dim=2)
            else:
                pad_video = head
        else:
            raise ValueError(f"Unsupported ref_pad_num={self.ref_pad_num} (expected -1, 0, or positive int)")

        full = torch.cat([cond_video, pad_video], dim=2)  # [1, 3, num_frames, H, W]
        y_latent = self._vae_encode_condition_video(full).to(dtype=dtype)  # [1, 16, T_lat, H_lat, W_lat]

        y = torch.cat([msk.unsqueeze(0).to(dtype=dtype), y_latent], dim=1)  # [1, 20, T_lat, H_lat, W_lat]
        return clip_feature, y

    @staticmethod
    def _pil_list_to_tensor(
        frames: list[PIL.Image.Image],
        *,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert a list of PIL frames to [1, 3, T, H, W] in [-1, 1]."""
        import numpy as np
        from PIL.Image import Resampling
        arrs = []
        for f in frames:
            if f.size != (width, height):
                f = f.resize((width, height), Resampling.BILINEAR)
            arrs.append(np.asarray(f.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
        t = torch.from_numpy(np.stack(arrs, axis=0)).to(device=device, dtype=torch.float32)  # [T, H, W, 3]
        t = t.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
        return t
