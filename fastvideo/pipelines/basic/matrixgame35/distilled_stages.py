# SPDX-License-Identifier: Apache-2.0
"""Composed stages for Matrix-Game 3.5 distilled first-person profiles."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.configs.pipelines.matrixgame35 import (
    matrixgame35_distilled_profile_settings,
    resolve_matrixgame35_hiar_scales,
)
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.basic.matrixgame35.camera import (
    RGB_FRAMES_PER_BLOCK,
    RGB_SUBFRAMES_PER_LATENT,
    MatrixGame35CameraTrajectory,
    build_prope_viewmats,
    load_camera_trajectory,
    normalize_matrixgame35_intrinsics,
)
from fastvideo.pipelines.basic.matrixgame35.base_stages import preprocess_matrixgame35_anchor
from fastvideo.pipelines.basic.matrixgame35.codec import (
    decode_matrixgame35_tiled_video,
    encode_matrixgame35_video,
    encode_matrixgame35_tiled_video,
    matrixgame35_tiled_memory_latents,
    matrixgame35_uint8_to_frames,
    matrixgame35_video_to_uint8,
)
from fastvideo.pipelines.basic.matrixgame35.causal_kv_cache import (
    causal_kv_frame_count,
    concat_causal_kv_caches,
    tail_causal_kv_cache_frames,
    trim_causal_kv_rolling_window,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_memory import (
    MatrixGame35DistilledPatchMemory,
    MatrixGame35DynamicContextEntry,
    MatrixGame35DynamicContextPool,
    is_da3_insufficient_non_sky_error,
)
from fastvideo.pipelines.basic.matrixgame35.distilled_profiles import (
    distilled_hiar_noise_seed,
    distilled_profile_guidance_scale,
    hiar_sde_corrupt_clean_latents,
    make_distilled_hiar_noise,
    trim_distilled_rolling_latents,
)
from fastvideo.pipelines.basic.matrixgame35.patch_memory import MatrixGame35DepthAdapter
from fastvideo.pipelines.basic.matrixgame35.prompts import (
    MATRIXGAME35_NEGATIVE_PROMPT,
    resolve_matrixgame35_section_prompts,
)
from fastvideo.pipelines.basic.matrixgame35.schedule import (
    MatrixGame35DistilledSchedule,
    RELEASED_DENOISING_IDS,
    build_distilled_schedule,
    distilled_noise_seeds,
    x0_renoise_transition,
)
from fastvideo.pipelines.basic.matrixgame35.runtime import (
    matrixgame35_autocast_context,
    move_matrixgame35_transformer_for_forward,
    offload_matrixgame35_transformer,
    run_matrixgame35_vae_operation,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.validators import StageValidators as V, VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

DISTILLED_DEFAULT_PROFILE = "standard"
DISTILLED_HEIGHT = 704
DISTILLED_WIDTH = 1280
DISTILLED_CHUNK_SIZE = 3
DISTILLED_CONTEXT_CHUNKS = 7
DISTILLED_GUIDANCE_SCALE = 3.0
DISTILLED_DEFAULT_SEED = 3407
MAX_TORCH_SEED = 2**64 - 1
DISTILLED_NEGATIVE_PROMPT = MATRIXGAME35_NEGATIVE_PROMPT


class MatrixGame35DistilledInputValidationStage(InputValidationStage):
    """Validate a released profile while preserving seed/block overrides."""

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = super().verify_input(batch, fastvideo_args)
        has_caption = isinstance(batch.caption_path, str) and bool(batch.caption_path)
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt) or V.list_not_empty(batch.prompt_embeds) or has_caption,
        )
        return result

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if (batch.height, batch.width) != (DISTILLED_HEIGHT, DISTILLED_WIDTH):
            raise ValueError(f"Matrix-Game 3.5 distilled inference requires {DISTILLED_HEIGHT}x{DISTILLED_WIDTH}.")
        if not isinstance(batch.num_frames, int) or batch.num_frames <= 1:
            raise ValueError("Matrix-Game 3.5 num_frames must be an integer greater than one.")
        if (batch.num_frames - 1) % RGB_FRAMES_PER_BLOCK:
            raise ValueError("Matrix-Game 3.5 requires num_frames = 1 + 84 * num_blocks, "
                             f"got {batch.num_frames}.")
        if batch.num_inference_steps != len(RELEASED_DENOISING_IDS):
            raise ValueError("Matrix-Game 3.5 distilled inference requires exactly three denoising steps.")
        config = fastvideo_args.pipeline_config
        profile = str(config.matrixgame35_distilled_profile)
        profile_settings = matrixgame35_distilled_profile_settings(profile)
        hiar_scales = resolve_matrixgame35_hiar_scales(
            profile,
            config.matrixgame35_distilled_hiar_scales,
            num_steps=len(RELEASED_DENOISING_IDS),
        )
        required_guidance = distilled_profile_guidance_scale(profile)
        if float(batch.guidance_scale) != required_guidance:
            raise ValueError(f"Matrix-Game 3.5 distilled profile {profile!r} requires "
                             f"guidance_scale={required_guidance}.")
        if batch.seed is None:
            batch.seed = DISTILLED_DEFAULT_SEED
        num_blocks = (batch.num_frames - 1) // RGB_FRAMES_PER_BLOCK
        total_chunks = num_blocks * DISTILLED_CONTEXT_CHUNKS
        final_chunk = total_chunks - 1
        max_seed_offset = final_chunk + 50000
        if profile_settings["prefix_noise_mode"] == "hiar_sde":
            max_seed_offset = max(
                max_seed_offset,
                distilled_hiar_noise_seed(
                    0,
                    batch_index=0,
                    chunk_index=final_chunk,
                    step_index=len(RELEASED_DENOISING_IDS) - 1,
                    dynamic_context=bool(profile_settings["noise_dynamic_context"]),
                ),
            )
        if (not isinstance(batch.seed, int) or isinstance(batch.seed, bool) or batch.seed < 0
                or batch.seed + max_seed_offset > MAX_TORCH_SEED):
            raise ValueError("Matrix-Game 3.5 seed and all derived distilled seeds must lie in "
                             f"[0, {MAX_TORCH_SEED}], got {batch.seed}.")
        if batch.num_videos_per_prompt != 1:
            raise ValueError("Matrix-Game 3.5 distilled rollout supports one video per prompt.")
        if int(getattr(fastvideo_args, "sp_size", 1)) != 1:
            raise ValueError("Matrix-Game 3.5 PRoPE currently requires sequence parallel size 1.")
        if batch.image_path is None and batch.pil_image is None:
            raise ValueError("Distilled first-person inference requires an anchor image.")
        if batch.camera_trajectory is None:
            raise ValueError("Distilled first-person inference requires camera_trajectory.")
        if batch.subject_ref_source is not None or batch.subject_ref_latents is not None:
            raise ValueError("Distilled first-person inference does not support subject references.")
        camera_path = Path(batch.camera_trajectory)
        if camera_path.suffix.lower() != ".npz" or not camera_path.is_file():
            raise ValueError("camera_trajectory must point to an existing official .npz file.")
        convention = str(getattr(batch, "camera_convention", "c2w") or "c2w").lower()
        if convention not in {"c2w", "w2c"}:
            raise ValueError("camera_convention must be 'c2w' or 'w2c'.")
        batch.camera_convention = convention
        requested_profile = batch.extra.get("matrixgame35_profile")
        if requested_profile is not None and requested_profile != profile:
            raise ValueError("matrixgame35_profile generation override must match the loaded pipeline config: "
                             f"{requested_profile!r} != {profile!r}.")
        if batch.negative_prompt not in (None, "", DISTILLED_NEGATIVE_PROMPT):
            raise ValueError("Matrix-Game 3.5 distilled inference uses the released fixed negative prompt.")
        batch.negative_prompt = DISTILLED_NEGATIVE_PROMPT
        batch.extra["matrixgame35_profile"] = profile
        batch.extra["matrixgame35_hiar_scales"] = hiar_scales
        batch.section_prompts = resolve_matrixgame35_section_prompts(
            batch.prompt,
            batch.section_prompts,
            batch.caption_path,
            num_frames=batch.num_frames,
        )

        preprocess_matrixgame35_anchor(batch, height=DISTILLED_HEIGHT, width=DISTILLED_WIDTH)
        prepared_anchor = batch.pil_image
        if isinstance(prepared_anchor, torch.Tensor):
            batch.pil_image = None
        original_prompt = batch.prompt
        if original_prompt is None:
            batch.prompt = batch.section_prompts[0]
        batch = super().forward(batch, fastvideo_args)
        batch.prompt = original_prompt
        if isinstance(prepared_anchor, torch.Tensor):
            batch.pil_image = prepared_anchor.to(self.device)
        if not isinstance(batch.pil_image, torch.Tensor) or batch.pil_image.shape != (
                1,
                3,
                1,
                DISTILLED_HEIGHT,
                DISTILLED_WIDTH,
        ):
            shape = None if not isinstance(batch.pil_image, torch.Tensor) else tuple(batch.pil_image.shape)
            raise ValueError(f"anchor preprocessing must produce [1,3,1,704,1280], got {shape}.")
        return batch


class MatrixGame35DistilledRolloutStage(PipelineStage):
    """Generate three latent frames at a time with bounded causal K/V memory."""

    performance_component_metric = "distilled_rollout_time_s"

    def __init__(
        self,
        transformer: Any,
        vae: Any,
        depth_adapter: MatrixGame35DepthAdapter | None,
        *,
        memory_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.depth_adapter = depth_adapter
        self.memory_factory = memory_factory or MatrixGame35DistilledPatchMemory
        self._uses_native_memory = memory_factory is None

    def set_depth_adapter(self, depth_adapter: MatrixGame35DepthAdapter | None) -> None:
        self.depth_adapter = depth_adapter

    def _encode_video(self, video: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        return run_matrixgame35_vae_operation(
            self.vae,
            video,
            precision=fastvideo_args.pipeline_config.vae_precision,
            device=get_local_torch_device(),
            fastvideo_args=fastvideo_args,
            operation=encode_matrixgame35_video,
        )

    def _decode_video(self, latents: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        config = fastvideo_args.pipeline_config
        return run_matrixgame35_vae_operation(
            self.vae,
            latents,
            precision=config.vae_decode_precision or config.vae_precision,
            device=get_local_torch_device(),
            fastvideo_args=fastvideo_args,
            operation=decode_matrixgame35_tiled_video,
        )

    def _memory_latents(self, frames: np.ndarray, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        normalized = matrixgame35_uint8_to_frames(frames, device="cpu")
        return run_matrixgame35_vae_operation(
            self.vae,
            normalized,
            precision=fastvideo_args.pipeline_config.vae_precision,
            device=get_local_torch_device(),
            fastvideo_args=fastvideo_args,
            operation=matrixgame35_tiled_memory_latents,
        )

    @staticmethod
    def _camera_info(
        trajectory: MatrixGame35CameraTrajectory,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        total_noisy_latents = (int(trajectory.c2w.shape[0]) - 1) // RGB_SUBFRAMES_PER_LATENT
        anchor_c2w = trajectory.c2w[:1].reshape(1, 1, 1, 4, 4).expand(1, 1, 4, 4, 4)
        anchor_K = trajectory.intrinsics[:1].reshape(1, 1, 1, 3, 3).expand(1, 1, 4, 3, 3)
        noisy_c2w = trajectory.c2w[1:].reshape(1, total_noisy_latents, 4, 4, 4)
        noisy_K = trajectory.intrinsics[1:].reshape(1, total_noisy_latents, 4, 3, 3)
        c2w = torch.cat((anchor_c2w, noisy_c2w), dim=1)
        w2c = torch.linalg.inv(c2w.double()).float().to(device=device, dtype=dtype)
        c2w = c2w.to(device=device, dtype=torch.float32)
        intrinsics = torch.cat((anchor_K, noisy_K), dim=1).to(device=device, dtype=torch.float32)
        viewmats = build_prope_viewmats(
            c2w,
            intrinsics,
            image_height=DISTILLED_HEIGHT,
            image_width=DISTILLED_WIDTH,
            translation_scale="logd4",
            dtype=dtype,
        )
        return w2c, viewmats

    def _transformer_forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        *,
        step_index: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = hidden_states.device
        dtype = hidden_states.dtype
        with matrixgame35_autocast_context(device, dtype, fastvideo_args.disable_autocast), set_forward_context(
                current_timestep=step_index,
                attn_metadata=None,
                forward_batch=batch,
        ):
            return self.transformer(hidden_states, context, timestep, **kwargs)

    def _append_memory(
        self,
        memory: Any,
        frames: np.ndarray,
        *,
        w2c: np.ndarray,
        intrinsics: np.ndarray,
        fastvideo_args: FastVideoArgs,
    ) -> np.ndarray:
        frames = np.ascontiguousarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"memory frames must be uint8 [T,H,W,3], got {frames.shape} {frames.dtype}.")
        memory.append(
            latents=self._memory_latents(frames, fastvideo_args),
            w2c=w2c,
            intrinsics=intrinsics,
            frames=list(frames),
            depth_adapter=self.depth_adapter,
        )
        return frames

    def _dynamic_context_entries(
        self,
        previous_rgb: np.ndarray,
        generated: np.ndarray,
        *,
        chunk_index: int,
        target_w2c: np.ndarray,
        fastvideo_args: FastVideoArgs,
    ) -> list[MatrixGame35DynamicContextEntry]:
        previous_rgb = np.asarray(previous_rgb)
        generated = np.asarray(generated)
        if previous_rgb.ndim != 3 or previous_rgb.shape[-1] != 3 or previous_rgb.dtype != np.uint8:
            raise ValueError("previous dynamic-context RGB must be one uint8 HWC frame.")
        if (generated.ndim != 4 or generated.shape[0] != DISTILLED_CHUNK_SIZE * RGB_SUBFRAMES_PER_LATENT
                or generated.shape[-1] != 3 or generated.dtype != np.uint8):
            raise ValueError("generated dynamic-context RGB must contain twelve uint8 HWC frames.")
        windows: list[np.ndarray] = []
        for local_index in range(DISTILLED_CHUNK_SIZE):
            start = local_index * RGB_SUBFRAMES_PER_LATENT
            previous = previous_rgb if local_index == 0 else generated[start - 1]
            windows.append(np.concatenate((previous[None], generated[start:start + 4]), axis=0))

        video_windows = torch.stack(
            [matrixgame35_uint8_to_frames(window, device="cpu").permute(1, 0, 2, 3) for window in windows])

        def _encode_windows(vae: Any, videos: torch.Tensor) -> torch.Tensor:
            encoded_windows = []
            for video in videos:
                encoded = encode_matrixgame35_tiled_video(vae, video.unsqueeze(0))
                if encoded.ndim != 5 or encoded.shape[0] != 1 or encoded.shape[2] < 1:
                    raise ValueError(f"dynamic context VAE encode returned {tuple(encoded.shape)}.")
                encoded_windows.append(encoded[:, :, -1:])
            return torch.cat(encoded_windows, dim=0)

        encoded = run_matrixgame35_vae_operation(
            self.vae,
            video_windows,
            precision=fastvideo_args.pipeline_config.vae_precision,
            device=get_local_torch_device(),
            fastvideo_args=fastvideo_args,
            operation=_encode_windows,
        )
        entries = []
        for local_index in range(DISTILLED_CHUNK_SIZE):
            noisy_index = 1 + chunk_index * DISTILLED_CHUNK_SIZE + local_index
            rgb_start = local_index * RGB_SUBFRAMES_PER_LATENT
            entries.append(
                MatrixGame35DynamicContextEntry(
                    latent=encoded[local_index:local_index + 1, :, -1:].detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    ).contiguous(),
                    position=1 + noisy_index,
                    camera_frame=noisy_index,
                    source_timeline_position=noisy_index,
                    representative_w2c=np.asarray(target_w2c[rgb_start + 1], dtype=np.float32),
                ))
        return entries

    def _build_hiar_prediction_cache(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        *,
        step_index: int,
        chunk_index: int,
        rolling_latents: torch.Tensor,
        rolling_positions: list[int],
        rolling_frames: list[int],
        selected_context: MatrixGame35DynamicContextEntry | None,
        clean_context_cache: list[dict[str, Any]] | None,
        context: torch.Tensor,
        schedule: MatrixGame35DistilledSchedule,
        hiar_scales: tuple[float, ...],
        noise_dynamic_context: bool,
        camera_info: Any,
    ) -> tuple[list[dict[str, Any]], list[int], list[int]]:
        """Rebuild one ephemeral HiAR prefix cache without mutating the clean cache."""
        device = rolling_latents.device
        dtype = rolling_latents.dtype
        next_timestep = (float(schedule.timesteps[step_index + 1]) if step_index + 1 < len(schedule.timesteps) else 0.0)
        next_sigma = schedule.sigmas[step_index + 1] if step_index + 1 < len(schedule.sigmas) else 0.0
        corruption_scale = hiar_scales[step_index]
        effective_timestep = next_timestep * corruption_scale
        rolling_seed = distilled_hiar_noise_seed(
            int(batch.seed),
            batch_index=0,
            chunk_index=chunk_index,
            step_index=step_index,
            dynamic_context=False,
        )
        if next_timestep <= 0.0 or corruption_scale <= 0.0:
            noised_rolling = rolling_latents.clone()
        else:
            rolling_noise = make_distilled_hiar_noise(rolling_latents, seed=rolling_seed)
            noised_rolling = hiar_sde_corrupt_clean_latents(
                rolling_latents,
                next_sigma,
                keep_first_clean=True,
                corruption_scale=corruption_scale,
                noise=rolling_noise,
            )
        rolling_timestep_frames = torch.full(
            (int(noised_rolling.shape[2]), ),
            effective_timestep,
            device=device,
            dtype=dtype,
        )
        rolling_timestep_frames[:1] = 0
        hiar_rolling_cache = self.transformer.init_causal_kv_caches()
        self._transformer_forward(
            batch,
            fastvideo_args,
            noised_rolling,
            context,
            rolling_timestep_frames,
            step_index=step_index,
            camera_info=camera_info,
            kv_caches=hiar_rolling_cache,
            current_positions=rolling_positions,
            current_frames=rolling_frames,
            current_cache_chunk_ids=[-1] * len(rolling_positions),
            write_cache=True,
        )

        hiar_context_cache = clean_context_cache
        if noise_dynamic_context and selected_context is not None:
            clean_dynamic_context = selected_context.latent.to(device=device, dtype=dtype)
            dynamic_seed = distilled_hiar_noise_seed(
                int(batch.seed),
                batch_index=0,
                chunk_index=chunk_index,
                step_index=step_index,
                dynamic_context=True,
            )
            if next_timestep <= 0.0 or corruption_scale <= 0.0:
                noised_dynamic_context = clean_dynamic_context.clone()
            else:
                dynamic_noise = make_distilled_hiar_noise(clean_dynamic_context, seed=dynamic_seed)
                noised_dynamic_context = hiar_sde_corrupt_clean_latents(
                    clean_dynamic_context,
                    next_sigma,
                    keep_first_clean=False,
                    corruption_scale=corruption_scale,
                    noise=dynamic_noise,
                )
            hiar_context_cache = self.transformer.init_causal_kv_caches()
            self._transformer_forward(
                batch,
                fastvideo_args,
                noised_dynamic_context,
                context,
                torch.full(
                    (int(noised_dynamic_context.shape[2]), ),
                    effective_timestep,
                    device=device,
                    dtype=dtype,
                ),
                step_index=step_index,
                camera_info=camera_info,
                kv_caches=hiar_context_cache,
                current_positions=[selected_context.position],
                current_frames=[selected_context.camera_frame],
                current_cache_chunk_ids=[chunk_index],
                write_cache=True,
            )
        hiar_cache = concat_causal_kv_caches(hiar_context_cache, hiar_rolling_cache)
        return (
            hiar_cache,
            list(hiar_cache[0].get("positions", [])),
            list(hiar_cache[0].get("frames", [])),
        )

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if self._uses_native_memory and self.depth_adapter is None:
            raise RuntimeError("Matrix-Game 3.5 distilled inference requires an explicit DA3 adapter/path.")
        if not isinstance(batch.pil_image, torch.Tensor):
            raise ValueError("distilled rollout requires the validated anchor tensor.")
        if not batch.prompt_embeds:
            raise ValueError("distilled rollout requires positive text embeddings.")
        if batch.camera_trajectory is None or not isinstance(batch.num_frames, int):
            raise ValueError("distilled rollout requires camera_trajectory and integer num_frames.")
        if (batch.num_frames - 1) % RGB_FRAMES_PER_BLOCK:
            raise ValueError("num_frames must equal 1 + 84 * num_blocks.")

        config = fastvideo_args.pipeline_config
        profile = str(config.matrixgame35_distilled_profile)
        profile_settings = matrixgame35_distilled_profile_settings(profile)
        guidance_scale = distilled_profile_guidance_scale(profile)
        uses_cfg = guidance_scale > 1.0
        if float(batch.guidance_scale) != guidance_scale:
            raise ValueError(f"Matrix-Game 3.5 distilled profile {profile!r} requires "
                             f"guidance_scale={guidance_scale}.")
        hiar_scales = resolve_matrixgame35_hiar_scales(
            profile,
            config.matrixgame35_distilled_hiar_scales,
            num_steps=len(RELEASED_DENOISING_IDS),
        )
        device = get_local_torch_device()
        dtype = PRECISION_TO_TYPE[config.dit_precision]
        if dtype != torch.bfloat16 and not fastvideo_args.disable_autocast:
            raise ValueError("The released distilled schedule requires BF16 DiT execution.")
        if getattr(self.transformer, "causal", True) is False:
            raise ValueError("Matrix-Game 3.5 distilled inference requires the causal DiT.")
        positive = batch.prompt_embeds[0].to(device=device, dtype=dtype)
        num_sections = (batch.num_frames - 1) // RGB_FRAMES_PER_BLOCK
        if positive.ndim != 3 or positive.shape[0] != num_sections:
            raise ValueError(f"distilled rollout requires one positive text embedding per section ({num_sections}).")
        negative = None
        if uses_cfg:
            if not batch.negative_prompt_embeds:
                raise ValueError("distilled CFG requires a negative text embedding.")
            negative = batch.negative_prompt_embeds[0].to(device=device, dtype=dtype)
            if negative.ndim != 3 or negative.shape[0] != 1 or negative.shape[1:] != positive.shape[1:]:
                raise ValueError("distilled CFG requires one shared aligned negative text embedding.")

        def section_context(section_index: int) -> torch.Tensor:
            section = positive[section_index:section_index + 1]
            return torch.cat((negative, section), dim=0) if negative is not None else section

        context = section_context(0)

        trajectory = load_camera_trajectory(
            batch.camera_trajectory,
            convention=str(getattr(batch, "camera_convention", "c2w") or "c2w"),
            frame_count=batch.num_frames,
        )
        trajectory_w2c = torch.linalg.inv(trajectory.c2w.double()).float().cpu().numpy()
        trajectory_K = normalize_matrixgame35_intrinsics(
            trajectory.intrinsics.cpu().numpy(),
            image_height=DISTILLED_HEIGHT,
            image_width=DISTILLED_WIDTH,
            mode="per_frame",
        )
        trajectory = MatrixGame35CameraTrajectory(
            c2w=trajectory.c2w,
            intrinsics=torch.from_numpy(trajectory_K),
        )
        camera_info = self._camera_info(trajectory, device=device, dtype=dtype)

        anchor_latent = self._encode_video(batch.pil_image, fastvideo_args).to(device=device, dtype=dtype)
        if anchor_latent.ndim != 5 or anchor_latent.shape[0] != 1 or anchor_latent.shape[2] != 1:
            raise ValueError(f"anchor VAE encode must produce [1,C,1,H,W], got {tuple(anchor_latent.shape)}.")
        channels = int(anchor_latent.shape[1])
        if channels != int(getattr(self.transformer, "in_channels", channels)):
            raise ValueError("anchor latent channels do not match the transformer.")
        anchor_decoded = self._decode_video(anchor_latent, fastvideo_args)
        if anchor_decoded.shape[2] != 1:
            raise ValueError("one anchor latent must decode to one RGB frame.")
        anchor_frames = matrixgame35_video_to_uint8(anchor_decoded)

        memory = self.memory_factory()
        memory_enabled = True
        try:
            self._append_memory(
                memory,
                anchor_frames,
                w2c=trajectory_w2c[:1],
                intrinsics=trajectory_K[:1],
                fastvideo_args=fastvideo_args,
            )
        except Exception as error:
            if not is_da3_insufficient_non_sky_error(error):
                raise
            memory_enabled = False
        memory_anchor_w2c = np.asarray(trajectory_w2c[0], dtype=np.float32)

        rolling_cache = self.transformer.init_causal_kv_caches()
        move_matrixgame35_transformer_for_forward(
            self.transformer,
            device=device,
            fastvideo_args=fastvideo_args,
        )
        try:
            self._transformer_forward(
                batch,
                fastvideo_args,
                anchor_latent,
                context,
                torch.zeros(1, device=device, dtype=dtype),
                step_index=0,
                camera_info=camera_info,
                kv_caches=rolling_cache,
                current_positions=[1],
                current_frames=[0],
                current_cache_chunk_ids=[-1],
                write_cache=True,
            )
        finally:
            offload_matrixgame35_transformer(
                self.transformer,
                fastvideo_args=fastvideo_args,
            )

        schedule = build_distilled_schedule(device=device, model_dtype=dtype)
        total_chunks = ((batch.num_frames - 1) // RGB_FRAMES_PER_BLOCK) * DISTILLED_CONTEXT_CHUNKS
        original_anchor_context = MatrixGame35DynamicContextEntry(
            latent=anchor_latent.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            position=1,
            camera_frame=0,
            source_timeline_position=0,
            representative_w2c=np.asarray(trajectory_w2c[0], dtype=np.float32),
            source="forced_original_anchor",
        )
        context_pool = MatrixGame35DynamicContextPool(
            pose_pool_size=5,
            original_anchor=original_anchor_context,
            force_original_anchor=bool(profile_settings["force_original_anchor"]),
        )
        rolling_latents = anchor_latent.detach().clone(
        ) if profile_settings["prefix_noise_mode"] == "hiar_sde" else None
        previous_anchor = anchor_latent.detach().clone()
        previous_rgb = np.ascontiguousarray(anchor_frames[-1])
        generated_latents_cpu: list[torch.Tensor] = []
        stats: dict[str, Any] = {
            "profile": profile,
            "initial_noise_seeds": [],
            "renoise_seeds": [],
            "cache_frame_counts": [],
            "context_positions": [],
            "context_sources": [],
            "memory_candidate_frame_ids": [],
            "memory_published_chunks": [],
            "registration_decoded_chunks": [],
            "memory_enabled_after_c0": memory_enabled,
            "hiar_prefix_noise": {
                "mode": str(profile_settings["prefix_noise_mode"]),
                "dynamic_context_noised": bool(profile_settings["noise_dynamic_context"]),
                "noise_scales_by_step": list(hiar_scales),
                "chunks": [],
            },
        }

        for chunk_index in range(total_chunks):
            context = section_context(chunk_index // DISTILLED_CONTEXT_CHUNKS)
            first_rgb = 1 + chunk_index * DISTILLED_CHUNK_SIZE * RGB_SUBFRAMES_PER_LATENT
            last_rgb = first_rgb + DISTILLED_CHUNK_SIZE * RGB_SUBFRAMES_PER_LATENT
            target_w2c = trajectory_w2c[first_rgb:last_rgb]
            target_K = trajectory_K[first_rgb:last_rgb]
            positions = [2 + chunk_index * DISTILLED_CHUNK_SIZE + offset for offset in range(DISTILLED_CHUNK_SIZE)]
            camera_frames = [1 + chunk_index * DISTILLED_CHUNK_SIZE + offset for offset in range(DISTILLED_CHUNK_SIZE)]
            rolling_positions = list(rolling_cache[0].get("positions", []))
            rolling_frames = list(rolling_cache[0].get("frames", []))
            if not rolling_positions or not rolling_frames:
                raise RuntimeError(f"rolling cache is empty before chunk {chunk_index}.")

            mosaic = None
            if memory_enabled:
                memory_result = memory.query(
                    anchor_w2c=memory_anchor_w2c,
                    query_w2c=target_w2c,
                    query_intrinsics=target_K,
                )
                mosaic = memory_result.latents.unsqueeze(0).to(device=device, dtype=dtype)
                if mosaic.shape[2] != DISTILLED_CHUNK_SIZE:
                    raise ValueError("distilled memory must return one mosaic latent per current latent.")
                stats["memory_candidate_frame_ids"].append(memory_result.candidate_frame_ids)
            else:
                stats["memory_candidate_frame_ids"].append(())

            selected_context = context_pool.select(
                target_w2c,
                chunk_index=chunk_index,
                exclude_position=int(rolling_positions[0]),
                exclude_camera_frame=int(rolling_frames[0]),
            )
            context_cache = None
            initial_seed, renoise_seed = distilled_noise_seeds(int(batch.seed), batch_index=0, chunk_index=chunk_index)
            stats["initial_noise_seeds"].append(initial_seed)
            stats["renoise_seeds"].append(renoise_seed)
            move_matrixgame35_transformer_for_forward(
                self.transformer,
                device=device,
                fastvideo_args=fastvideo_args,
            )
            try:
                if selected_context is not None:
                    context_cache = self.transformer.init_causal_kv_caches()
                    context_latent = selected_context.latent.to(device=device, dtype=dtype)
                    self._transformer_forward(
                        batch,
                        fastvideo_args,
                        context_latent,
                        context,
                        torch.zeros(1, device=device, dtype=dtype),
                        step_index=0,
                        camera_info=camera_info,
                        kv_caches=context_cache,
                        current_positions=[selected_context.position],
                        current_frames=[selected_context.camera_frame],
                        current_cache_chunk_ids=[chunk_index],
                        write_cache=True,
                    )
                    stats["context_positions"].append(selected_context.position)
                    stats["context_sources"].append(selected_context.source)
                else:
                    stats["context_positions"].append(None)
                    stats["context_sources"].append(None)

                read_cache = concat_causal_kv_caches(context_cache, rolling_cache)
                cache_positions = list(read_cache[0].get("positions", []))
                cache_frames = list(read_cache[0].get("frames", []))
                current = torch.randn(
                    (1, channels, DISTILLED_CHUNK_SIZE, anchor_latent.shape[-2], anchor_latent.shape[-1]),
                    generator=torch.Generator("cpu").manual_seed(initial_seed),
                    dtype=torch.float32,
                    device="cpu",
                ).to(device=device, dtype=dtype)
                renoise_generator = torch.Generator("cpu").manual_seed(renoise_seed)
                mosaic_count = 0 if mosaic is None else int(mosaic.shape[2])

                if profile_settings["prefix_noise_mode"] == "hiar_sde":
                    assert rolling_latents is not None
                    stats["hiar_prefix_noise"]["chunks"].append({
                        "chunk_index":
                        chunk_index,
                        "rolling_prefix_frames":
                        int(rolling_latents.shape[2]),
                        "dynamic_context_frames":
                        int(selected_context is not None),
                        "context_timesteps": [
                            float(schedule.timesteps[index + 1]) if index + 1 < len(schedule.timesteps) else 0.0
                            for index in range(len(schedule.timesteps))
                        ],
                        "effective_context_timesteps":
                        [(float(schedule.timesteps[index + 1]) * hiar_scales[index] if index +
                          1 < len(schedule.timesteps) else 0.0) for index in range(len(schedule.timesteps))],
                        "rolling_noise_seeds": [
                            distilled_hiar_noise_seed(
                                int(batch.seed),
                                batch_index=0,
                                chunk_index=chunk_index,
                                step_index=index,
                                dynamic_context=False,
                            ) for index in range(len(schedule.timesteps))
                        ],
                        "dynamic_context_noise_seeds": [
                            distilled_hiar_noise_seed(
                                int(batch.seed),
                                batch_index=0,
                                chunk_index=chunk_index,
                                step_index=index,
                                dynamic_context=True,
                            ) for index in range(len(schedule.timesteps))
                        ] if selected_context is not None else [],
                    })

                for step_index, timestep in enumerate(schedule.timesteps):
                    prediction_cache = read_cache
                    prediction_cache_positions = cache_positions
                    prediction_cache_frames = cache_frames
                    if profile_settings["prefix_noise_mode"] == "hiar_sde":
                        assert rolling_latents is not None
                        prediction_cache, prediction_cache_positions, prediction_cache_frames = (
                            self._build_hiar_prediction_cache(
                                batch,
                                fastvideo_args,
                                step_index=step_index,
                                chunk_index=chunk_index,
                                rolling_latents=rolling_latents,
                                rolling_positions=rolling_positions,
                                rolling_frames=rolling_frames,
                                selected_context=selected_context,
                                clean_context_cache=context_cache,
                                context=context,
                                schedule=schedule,
                                hiar_scales=hiar_scales,
                                noise_dynamic_context=bool(profile_settings["noise_dynamic_context"]),
                                camera_info=camera_info,
                            ))
                    timestep_frames = torch.cat((
                        torch.zeros(mosaic_count, device=device, dtype=dtype),
                        torch.full(
                            (DISTILLED_CHUNK_SIZE, ),
                            float(timestep),
                            device=device,
                            dtype=dtype,
                        ),
                    ))
                    prediction = self._transformer_forward(
                        batch,
                        fastvideo_args,
                        current,
                        context,
                        timestep_frames,
                        step_index=step_index,
                        camera_info=camera_info,
                        kv_caches=prediction_cache,
                        mosaic_latents=mosaic,
                        current_positions=positions,
                        current_frames=camera_frames,
                        mosaic_positions=positions if mosaic is not None else None,
                        mosaic_frames=camera_frames if mosaic is not None else None,
                        cache_positions=prediction_cache_positions,
                        cache_frames=prediction_cache_frames,
                        cache_read_chunk_id=chunk_index,
                        write_cache=False,
                    )
                    if uses_cfg:
                        if prediction.shape[0] != 2:
                            raise ValueError("distilled CFG output must contain negative and positive batches.")
                        negative_prediction, positive_prediction = prediction.chunk(2, dim=0)
                        velocity = negative_prediction + guidance_scale * (positive_prediction - negative_prediction)
                    else:
                        if prediction.shape[0] != 1:
                            raise ValueError("distilled non-CFG output must contain one positive batch.")
                        velocity = prediction
                    if step_index + 1 < len(schedule.timesteps):
                        renoise = torch.randn(
                            current.shape,
                            generator=renoise_generator,
                            dtype=torch.float32,
                            device="cpu",
                        ).to(device=device, dtype=dtype)
                        current = x0_renoise_transition(
                            current,
                            velocity,
                            schedule.sigmas[step_index],
                            next_sigma=schedule.sigmas[step_index + 1],
                            renoise=renoise,
                        )
                    else:
                        current = x0_renoise_transition(
                            current,
                            velocity,
                            schedule.sigmas[step_index],
                        )
                generated = current

                self._transformer_forward(
                    batch,
                    fastvideo_args,
                    generated,
                    context,
                    torch.zeros(mosaic_count + DISTILLED_CHUNK_SIZE, device=device, dtype=dtype),
                    step_index=len(schedule.timesteps) - 1,
                    camera_info=camera_info,
                    kv_caches=read_cache,
                    mosaic_latents=mosaic,
                    current_positions=positions,
                    current_frames=camera_frames,
                    mosaic_positions=positions if mosaic is not None else None,
                    mosaic_frames=camera_frames if mosaic is not None else None,
                    cache_positions=cache_positions,
                    cache_frames=cache_frames,
                    cache_read_chunk_id=chunk_index,
                    write_cache=True,
                )
                current_cache = tail_causal_kv_cache_frames(
                    read_cache,
                    DISTILLED_CHUNK_SIZE,
                    context=f"distilled generated chunk {chunk_index}",
                )
                rolling_cache = trim_causal_kv_rolling_window(
                    concat_causal_kv_caches(rolling_cache, current_cache),
                    frames_per_chunk=DISTILLED_CHUNK_SIZE,
                    window_chunks=DISTILLED_CONTEXT_CHUNKS,
                )
                if rolling_latents is not None:
                    rolling_latents = trim_distilled_rolling_latents(
                        torch.cat((rolling_latents, generated.detach()), dim=2).contiguous(),
                        frames_per_chunk=DISTILLED_CHUNK_SIZE,
                        window_chunks=DISTILLED_CONTEXT_CHUNKS,
                    )
                    if int(rolling_latents.shape[2]) != causal_kv_frame_count(rolling_cache):
                        raise RuntimeError("HiAR rolling latent/cache provenance diverged after "
                                           f"chunk {chunk_index}: latents={int(rolling_latents.shape[2])}, "
                                           f"cache={causal_kv_frame_count(rolling_cache)}.")
            finally:
                offload_matrixgame35_transformer(
                    self.transformer,
                    fastvideo_args=fastvideo_args,
                )
            stats["cache_frame_counts"].append(causal_kv_frame_count(rolling_cache))
            generated_latents_cpu.append(generated.detach().to(device="cpu"))

            has_future = chunk_index + 1 < total_chunks
            accepts_generated_context = not bool(profile_settings["force_original_anchor"])
            if has_future and (accepts_generated_context or memory_enabled):
                decoded_prefix = self._decode_video(
                    torch.cat((previous_anchor, generated), dim=2),
                    fastvideo_args,
                )
                expected_rgb = 1 + DISTILLED_CHUNK_SIZE * RGB_SUBFRAMES_PER_LATENT
                if decoded_prefix.shape[2] != expected_rgb:
                    raise ValueError(f"one anchor + three latents must decode to {expected_rgb} RGB frames.")
                generated_frames = matrixgame35_video_to_uint8(decoded_prefix[:, :, 1:])
                stats["registration_decoded_chunks"].append(chunk_index)
                if accepts_generated_context:
                    context_pool.publish(
                        self._dynamic_context_entries(
                            previous_rgb,
                            generated_frames,
                            chunk_index=chunk_index,
                            target_w2c=target_w2c,
                            fastvideo_args=fastvideo_args,
                        ))
                if memory_enabled:
                    try:
                        self._append_memory(
                            memory,
                            generated_frames,
                            w2c=target_w2c,
                            intrinsics=target_K,
                            fastvideo_args=fastvideo_args,
                        )
                    except Exception as error:
                        if not is_da3_insufficient_non_sky_error(error):
                            raise
                    else:
                        stats["memory_published_chunks"].append(chunk_index)
                        memory_anchor_w2c = np.asarray(target_w2c[-1], dtype=np.float32)
                if accepts_generated_context:
                    previous_rgb = np.ascontiguousarray(generated_frames[-1])
            # Keep only the one-frame rolling anchors alive between chunks.  A
            # detached view would retain the full latent chunk storage.
            previous_anchor = generated[:, :, -1:].detach().clone()

        batch.latents = torch.cat((anchor_latent.detach().cpu(), *generated_latents_cpu), dim=2)
        batch.raw_latent_shape = tuple(batch.latents.shape)
        batch.output = self._decode_video(batch.latents, fastvideo_args).detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if batch.output.shape[2] != batch.num_frames:
            raise RuntimeError(
                f"distilled rollout published {batch.output.shape[2]} frames, expected {batch.num_frames}.")
        batch.extra["matrixgame35_distilled_stats"] = stats
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, lambda value: isinstance(value, torch.Tensor))
        result.add_check("prompt_embeds", batch.prompt_embeds,
                         lambda value: isinstance(value, list) and len(value) == 1)
        profile = str(fastvideo_args.pipeline_config.matrixgame35_distilled_profile)
        if distilled_profile_guidance_scale(profile) > 1.0:
            result.add_check(
                "negative_prompt_embeds",
                batch.negative_prompt_embeds,
                lambda value: isinstance(value, list) and len(value) == 1,
            )
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "output",
            batch.output,
            lambda value: isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.ndim == 5 and value.
            shape[2] == batch.num_frames,
        )
        return result


__all__ = [
    "DISTILLED_CHUNK_SIZE",
    "DISTILLED_CONTEXT_CHUNKS",
    "DISTILLED_DEFAULT_SEED",
    "DISTILLED_GUIDANCE_SCALE",
    "DISTILLED_HEIGHT",
    "DISTILLED_NEGATIVE_PROMPT",
    "DISTILLED_DEFAULT_PROFILE",
    "DISTILLED_WIDTH",
    "MatrixGame35DistilledInputValidationStage",
    "MatrixGame35DistilledRolloutStage",
]
