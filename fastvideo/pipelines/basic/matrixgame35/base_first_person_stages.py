# SPDX-License-Identifier: Apache-2.0
"""Stages for Matrix-Game 3.5 Base STANDARD rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.vision_utils import load_image, load_video
from fastvideo.pipelines.basic.matrixgame35.camera import (
    RGB_FRAMES_PER_BLOCK,
    RGB_SUBFRAMES_PER_LATENT,
    MatrixGame35CameraTrajectory,
    build_prope_viewmats,
    load_camera_trajectory,
    normalize_matrixgame35_intrinsics,
)
from fastvideo.pipelines.basic.matrixgame35.codec import (
    decode_matrixgame35_video,
    encode_matrixgame35_video,
    matrixgame35_memory_latents,
    matrixgame35_uint8_to_frames,
    matrixgame35_video_to_uint8,
)
from fastvideo.pipelines.basic.matrixgame35.layout import build_noncausal_latent_layout
from fastvideo.pipelines.basic.matrixgame35.patch_memory import (
    MatrixGame35BasePatchMemory,
    MatrixGame35DepthAdapter,
)
from fastvideo.pipelines.basic.matrixgame35.prompts import (
    MATRIXGAME35_NEGATIVE_PROMPT,
    resolve_matrixgame35_section_prompts,
)
from fastvideo.pipelines.basic.matrixgame35.schedule import (
    BASE_FLOW_SHIFT,
    base_flow_step,
    build_base_schedule,
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
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

BASE_GUIDANCE_SCALE = 5.0
BASE_SEED = 3407
LATENTS_PER_BLOCK = RGB_FRAMES_PER_BLOCK // RGB_SUBFRAMES_PER_LATENT
MAX_TORCH_SEED = 2**64 - 1


def preprocess_matrixgame35_anchor(batch: ForwardBatch, *, height: int, width: int) -> None:
    """Apply the released crop-before-resize anchor transform."""
    if batch.image_path is not None:
        batch.pil_image = load_video(batch.image_path)[0] if batch.image_path.endswith(".mp4") else load_image(
            batch.image_path)
        batch.image_path = None
    if not isinstance(batch.pil_image, Image.Image):
        return
    image = np.asarray(batch.pil_image.convert("RGB"))
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Matrix-Game 3.5 anchor has invalid size {source_width}x{source_height}.")
    crop_height = min(source_height, height)
    crop_width = min(source_width, width)
    top = max((source_height - crop_height) // 2, 0)
    left = max((source_width - crop_width) // 2, 0)
    image = np.array(image[top:top + crop_height, left:left + crop_width], copy=True, order="C")
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    tensor.mul_(2.0 / 255.0).sub_(1.0)
    if tensor.shape[-2:] != (height, width):
        tensor = F.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    batch.pil_image = tensor.unsqueeze(2)


def matrixgame35_base_block_seed(base_seed: int, *, batch_index: int, block_index: int) -> int:
    """Return the released independent noise stream for one Base block."""
    return int(base_seed) + int(batch_index) * 1000 + int(block_index)


class MatrixGame35BaseInputValidationStage(InputValidationStage):
    """Validate and materialize the released Base first-person inputs."""

    allow_subject_refs = False

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        config = fastvideo_args.pipeline_config
        expected_height = int(config.matrixgame35_height)
        expected_width = int(config.matrixgame35_width)
        if (batch.height, batch.width) != (expected_height, expected_width):
            raise ValueError("Matrix-Game 3.5 Base STANDARD requires "
                             f"{expected_height}x{expected_width}, got {batch.height}x{batch.width}.")
        if not isinstance(batch.num_frames, int) or batch.num_frames <= 1:
            raise ValueError("Matrix-Game 3.5 num_frames must be an integer greater than one.")
        if (batch.num_frames - 1) % RGB_FRAMES_PER_BLOCK:
            raise ValueError("Matrix-Game 3.5 requires num_frames = 1 + 84 * num_blocks, "
                             f"got {batch.num_frames}.")
        if (not isinstance(batch.num_inference_steps, int) or isinstance(batch.num_inference_steps, bool)
                or batch.num_inference_steps <= 0):
            raise ValueError("Matrix-Game 3.5 num_inference_steps must be a positive integer, "
                             f"got {batch.num_inference_steps}.")
        if (isinstance(batch.guidance_scale, bool) or not isinstance(batch.guidance_scale, int | float)
                or not np.isfinite(batch.guidance_scale) or batch.guidance_scale <= 0.0):
            raise ValueError("Matrix-Game 3.5 guidance_scale must be finite and positive, "
                             f"got {batch.guidance_scale}.")
        if batch.seed is None:
            batch.seed = BASE_SEED
        num_blocks = (batch.num_frames - 1) // RGB_FRAMES_PER_BLOCK
        if (not isinstance(batch.seed, int) or isinstance(batch.seed, bool) or batch.seed < 0 or
                matrixgame35_base_block_seed(batch.seed, batch_index=0, block_index=num_blocks - 1) > MAX_TORCH_SEED):
            raise ValueError("Matrix-Game 3.5 seed and all derived block seeds must lie in "
                             f"[0, {MAX_TORCH_SEED}], got {batch.seed}.")
        if batch.num_videos_per_prompt != 1:
            raise ValueError("Matrix-Game 3.5 Base rollout currently supports num_videos_per_prompt=1 only.")
        if getattr(fastvideo_args, "sp_size", 1) != 1:
            raise ValueError("Matrix-Game 3.5 PRoPE currently requires sequence parallel size 1.")
        if batch.image_path is None and batch.pil_image is None:
            raise ValueError("Matrix-Game 3.5 Base requires an anchor image via image_path or pil_image.")
        if batch.camera_trajectory is None:
            raise ValueError("Matrix-Game 3.5 Base requires camera_trajectory pointing to the official .npz format.")
        if batch.subject_ref_source is not None and not self.allow_subject_refs:
            raise ValueError("Matrix-Game 3.5 Base first-person does not accept subject_ref_source; "
                             "use the Base third-person variant instead.")
        if batch.subject_ref_latents is not None and not self.allow_subject_refs:
            raise ValueError("Matrix-Game 3.5 Base first-person does not accept direct subject_ref_latents; "
                             "use the Base third-person variant instead.")
        if batch.camera_convention not in ("c2w", "w2c"):
            raise ValueError("Matrix-Game 3.5 camera_convention must be 'c2w' or 'w2c', "
                             f"got {batch.camera_convention!r}.")
        if batch.negative_prompt not in (None, "", MATRIXGAME35_NEGATIVE_PROMPT):
            raise ValueError("Matrix-Game 3.5 Base uses the released fixed negative prompt.")
        batch.negative_prompt = MATRIXGAME35_NEGATIVE_PROMPT
        batch.section_prompts = resolve_matrixgame35_section_prompts(
            batch.prompt,
            batch.section_prompts,
            batch.caption_path,
            num_frames=batch.num_frames,
        )
        camera_path = Path(batch.camera_trajectory)
        if camera_path.suffix.lower() != ".npz":
            raise ValueError("Matrix-Game 3.5 camera_trajectory must use the official .npz format.")
        if not camera_path.is_file():
            raise FileNotFoundError(f"Matrix-Game 3.5 camera trajectory does not exist: {camera_path}")

        preprocess_matrixgame35_anchor(batch, height=expected_height, width=expected_width)
        prepared_anchor = batch.pil_image
        if isinstance(prepared_anchor, torch.Tensor):
            batch.pil_image = None
        original_prompt = batch.prompt
        if original_prompt is None:
            batch.prompt = batch.section_prompts[0]
        batch.do_classifier_free_guidance = float(batch.guidance_scale) != 1.0
        batch = super().forward(batch, fastvideo_args)
        batch.prompt = original_prompt
        if isinstance(prepared_anchor, torch.Tensor):
            batch.pil_image = prepared_anchor.to(self.device)
        if not isinstance(batch.pil_image, torch.Tensor) or batch.pil_image.shape != (
                1,
                3,
                1,
                expected_height,
                expected_width,
        ):
            shape = None if not isinstance(batch.pil_image, torch.Tensor) else tuple(batch.pil_image.shape)
            raise ValueError("Matrix-Game 3.5 anchor preprocessing must produce "
                             f"[1,3,1,{expected_height},{expected_width}], got {shape}.")
        return batch

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = super().verify_output(batch, fastvideo_args)
        result.add_check(
            "camera_trajectory",
            batch.camera_trajectory,
            lambda value: isinstance(value, str) and value.endswith(".npz"),
        )
        return result


class MatrixGame35BaseRolloutStage(PipelineStage):
    """Generate Base-standard 84-frame sections with Patch Memory."""

    performance_component_metric = "base_rollout_time_s"

    def __init__(self, transformer: Any, vae: Any, depth_adapter: MatrixGame35DepthAdapter | None) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.depth_adapter = depth_adapter

    def set_depth_adapter(self, depth_adapter: MatrixGame35DepthAdapter | None) -> None:
        """Replace the lazy production adapter with an injected implementation."""
        self.depth_adapter = depth_adapter

    def _run_vae(
        self,
        value: torch.Tensor,
        *,
        precision: str,
        fastvideo_args: FastVideoArgs,
        operation: Any,
    ) -> torch.Tensor:
        return run_matrixgame35_vae_operation(
            self.vae,
            value,
            precision=precision,
            device=get_local_torch_device(),
            fastvideo_args=fastvideo_args,
            operation=operation,
        )

    def _encode_video(self, video: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        return self._run_vae(
            video,
            precision=fastvideo_args.pipeline_config.vae_precision,
            fastvideo_args=fastvideo_args,
            operation=encode_matrixgame35_video,
        )

    def _decode_video(self, latents: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        precision = fastvideo_args.pipeline_config.vae_decode_precision
        return self._run_vae(
            latents,
            precision=precision or fastvideo_args.pipeline_config.vae_precision,
            fastvideo_args=fastvideo_args,
            operation=decode_matrixgame35_video,
        )

    def _memory_latents(self, frames: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        return self._run_vae(
            frames,
            precision=fastvideo_args.pipeline_config.vae_precision,
            fastvideo_args=fastvideo_args,
            operation=matrixgame35_memory_latents,
        )

    def _move_transformer_for_forward(self, device: torch.device, fastvideo_args: FastVideoArgs) -> None:
        move_matrixgame35_transformer_for_forward(
            self.transformer,
            device=device,
            fastvideo_args=fastvideo_args,
        )

    def _offload_transformer(self, fastvideo_args: FastVideoArgs) -> None:
        offload_matrixgame35_transformer(
            self.transformer,
            fastvideo_args=fastvideo_args,
        )

    @staticmethod
    def _trajectory_w2c(trajectory: MatrixGame35CameraTrajectory) -> np.ndarray:
        c2w = trajectory.c2w.cpu().numpy().astype(np.float64)
        return np.ascontiguousarray(np.linalg.inv(c2w).astype(np.float32))

    @staticmethod
    def _build_camera_info(
        *,
        clean_w2c: np.ndarray,
        clean_intrinsics: np.ndarray,
        target_w2c: np.ndarray,
        target_intrinsics: np.ndarray,
        mosaic_indices: torch.Tensor,
        image_height: int,
        image_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        target_w2c_tensor = torch.from_numpy(np.ascontiguousarray(target_w2c)).reshape(
            1,
            LATENTS_PER_BLOCK,
            RGB_SUBFRAMES_PER_LATENT,
            4,
            4,
        )
        target_c2w = torch.linalg.inv(target_w2c_tensor.double()).float()
        target_intrinsics_tensor = torch.from_numpy(np.ascontiguousarray(target_intrinsics)).reshape(
            1,
            LATENTS_PER_BLOCK,
            RGB_SUBFRAMES_PER_LATENT,
            3,
            3,
        )

        clean_w2c = np.asarray(clean_w2c, dtype=np.float32)
        clean_intrinsics = np.asarray(clean_intrinsics, dtype=np.float32)
        if clean_w2c.shape == (4, 4):
            clean_w2c = np.repeat(clean_w2c[None], RGB_SUBFRAMES_PER_LATENT, axis=0)
        if clean_intrinsics.shape == (3, 3):
            clean_intrinsics = np.repeat(clean_intrinsics[None], RGB_SUBFRAMES_PER_LATENT, axis=0)
        if clean_w2c.shape != (RGB_SUBFRAMES_PER_LATENT, 4, 4):
            raise ValueError(f"clean_w2c must contain one or four RGB poses, got {clean_w2c.shape}.")
        if clean_intrinsics.shape != (RGB_SUBFRAMES_PER_LATENT, 3, 3):
            raise ValueError(f"clean_intrinsics must contain one or four RGB intrinsics, got {clean_intrinsics.shape}.")

        clean_w2c_tensor = torch.from_numpy(np.ascontiguousarray(clean_w2c)).reshape(
            1,
            1,
            RGB_SUBFRAMES_PER_LATENT,
            4,
            4,
        )
        clean_c2w = torch.linalg.inv(clean_w2c_tensor.double()).float()
        clean_intrinsics_tensor = torch.from_numpy(np.ascontiguousarray(clean_intrinsics)).reshape(
            1,
            1,
            RGB_SUBFRAMES_PER_LATENT,
            3,
            3,
        )

        indices = mosaic_indices.to(device="cpu", dtype=torch.long)
        mosaic_c2w = target_c2w.index_select(1, indices)
        mosaic_w2c = target_w2c_tensor.index_select(1, indices)
        mosaic_intrinsics = target_intrinsics_tensor.index_select(1, indices)
        full_c2w = torch.cat((clean_c2w, mosaic_c2w, target_c2w), dim=1).to(device=device, dtype=torch.float32)
        full_w2c = torch.cat((clean_w2c_tensor, mosaic_w2c, target_w2c_tensor), dim=1).to(
            device=device,
            dtype=dtype,
        )
        full_intrinsics = torch.cat((clean_intrinsics_tensor, mosaic_intrinsics, target_intrinsics_tensor),
                                    dim=1).to(device=device, dtype=torch.float32)
        viewmats = build_prope_viewmats(
            full_c2w,
            full_intrinsics,
            image_height=image_height,
            image_width=image_width,
            translation_scale=50.0,
            dtype=dtype,
        )
        return full_w2c, viewmats

    def _append_memory_frames(
        self,
        memory: MatrixGame35BasePatchMemory,
        decoded: torch.Tensor,
        *,
        w2c: np.ndarray,
        intrinsics: np.ndarray,
        fastvideo_args: FastVideoArgs,
    ) -> np.ndarray:
        if self.depth_adapter is None:
            raise RuntimeError("Matrix-Game 3.5 Base STANDARD requires a configured depth adapter for Patch Memory.")
        uint8_frames = matrixgame35_video_to_uint8(decoded)
        normalized_frames = matrixgame35_uint8_to_frames(uint8_frames, device=decoded.device)
        memory.append(
            latents=self._memory_latents(normalized_frames, fastvideo_args),
            w2c=w2c,
            intrinsics=intrinsics,
            frames=list(uint8_frames),
            depth_adapter=self.depth_adapter,
        )
        return uint8_frames

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if self.depth_adapter is None:
            raise RuntimeError("Matrix-Game 3.5 Base STANDARD requires a configured depth adapter for Patch Memory.")
        if not isinstance(batch.pil_image, torch.Tensor):
            raise ValueError("Matrix-Game 3.5 rollout requires the validated anchor tensor.")
        if not batch.prompt_embeds:
            raise ValueError("Matrix-Game 3.5 Base requires positive text embeddings.")
        do_classifier_free_guidance = float(batch.guidance_scale) != 1.0
        if do_classifier_free_guidance and not batch.negative_prompt_embeds:
            raise ValueError("Matrix-Game 3.5 Base CFG requires negative text embeddings when guidance_scale != 1.")
        if batch.camera_trajectory is None or not isinstance(batch.num_frames, int):
            raise ValueError("Matrix-Game 3.5 rollout requires camera_trajectory and integer num_frames.")
        if (batch.num_frames - 1) % RGB_FRAMES_PER_BLOCK:
            raise ValueError("Matrix-Game 3.5 requires num_frames = 1 + 84 * num_blocks.")

        config = fastvideo_args.pipeline_config
        device = get_local_torch_device()
        dtype = PRECISION_TO_TYPE[config.dit_precision]
        num_blocks = (batch.num_frames - 1) // RGB_FRAMES_PER_BLOCK
        positive_prompts = batch.prompt_embeds[0].to(device=device, dtype=dtype)
        negative_prompt = None
        if do_classifier_free_guidance:
            negative_prompt = batch.negative_prompt_embeds[0].to(device=device, dtype=dtype)
        if positive_prompts.ndim != 3 or positive_prompts.shape[0] != num_blocks:
            raise ValueError(f"Matrix-Game 3.5 Base requires one positive text embedding per section ({num_blocks}).")
        if (negative_prompt is not None and (negative_prompt.ndim != 3 or negative_prompt.shape[0] != 1
                                             or negative_prompt.shape[1:] != positive_prompts.shape[1:])):
            raise ValueError("Matrix-Game 3.5 Base CFG requires one shared aligned negative text embedding.")

        trajectory = load_camera_trajectory(
            batch.camera_trajectory,
            convention=batch.camera_convention,
            frame_count=batch.num_frames,
        )
        trajectory_w2c = self._trajectory_w2c(trajectory)
        intrinsics = normalize_matrixgame35_intrinsics(
            trajectory.intrinsics.cpu().numpy(),
            image_height=int(config.matrixgame35_height),
            image_width=int(config.matrixgame35_width),
            mode="first_frame",
        )
        anchor_latent = self._encode_video(batch.pil_image, fastvideo_args).to(device=device, dtype=dtype)
        if anchor_latent.ndim != 5 or anchor_latent.shape[0] != 1 or anchor_latent.shape[2] != 1:
            raise ValueError(f"anchor VAE encode must produce [1,C,1,H,W], got {tuple(anchor_latent.shape)}.")
        expected_channels = int(getattr(self.transformer, "in_channels", anchor_latent.shape[1]))
        if anchor_latent.shape[1] != expected_channels:
            raise ValueError(
                f"anchor VAE encode produced {anchor_latent.shape[1]} channels; transformer expects {expected_channels}."
            )

        anchor_decoded = self._decode_video(anchor_latent, fastvideo_args)
        if anchor_decoded.shape[2] != 1:
            raise ValueError(f"one anchor latent must decode to one RGB frame, got {anchor_decoded.shape[2]}.")
        memory = MatrixGame35BasePatchMemory()
        self._append_memory_frames(
            memory,
            anchor_decoded,
            w2c=trajectory_w2c[:1],
            intrinsics=intrinsics[:1],
            fastvideo_args=fastvideo_args,
        )

        schedule = build_base_schedule(
            num_inference_steps=batch.num_inference_steps,
            shift=float(BASE_FLOW_SHIFT if config.flow_shift is None else config.flow_shift),
            device=device,
        )
        generated_chunks: list[torch.Tensor] = []
        del anchor_decoded
        current_clean = anchor_latent

        for block_index in range(num_blocks):
            prompt = positive_prompts[block_index:block_index + 1]
            first_target = 1 + block_index * RGB_FRAMES_PER_BLOCK
            last_target = first_target + RGB_FRAMES_PER_BLOCK
            raw_anchor_w2c = trajectory_w2c[first_target - 1]
            raw_target_w2c = trajectory_w2c[first_target:last_target]
            target_intrinsics = intrinsics[first_target:last_target]
            memory_result = memory.query(
                anchor_w2c=raw_anchor_w2c,
                query_w2c=raw_target_w2c,
                query_intrinsics=target_intrinsics,
            )
            mosaic = memory_result.latents.unsqueeze(0).to(device=device, dtype=dtype)
            mosaic_indices = torch.arange(LATENTS_PER_BLOCK, device=device, dtype=torch.long)
            if block_index == 0:
                clean_w2c = memory.w2c[-1]
                clean_intrinsics = memory.intrinsics[-1]
            else:
                clean_w2c = memory.w2c[-RGB_SUBFRAMES_PER_LATENT:]
                clean_intrinsics = memory.intrinsics[-RGB_SUBFRAMES_PER_LATENT:]
            camera_info = self._build_camera_info(
                clean_w2c=clean_w2c,
                clean_intrinsics=clean_intrinsics,
                target_w2c=memory_result.aligned_query_w2c,
                target_intrinsics=target_intrinsics,
                mosaic_indices=mosaic_indices,
                image_height=int(config.matrixgame35_height),
                image_width=int(config.matrixgame35_width),
                device=device,
                dtype=dtype,
            )

            generator = torch.Generator("cpu").manual_seed(
                matrixgame35_base_block_seed(
                    int(batch.seed),
                    batch_index=0,
                    block_index=block_index,
                ))
            noisy = torch.randn(
                (1, expected_channels, LATENTS_PER_BLOCK, anchor_latent.shape[-2], anchor_latent.shape[-1]),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=device, dtype=dtype)

            self._move_transformer_for_forward(device, fastvideo_args)
            try:
                for step_index, timestep in enumerate(schedule.timesteps):
                    layout = build_noncausal_latent_layout(
                        noisy,
                        timestep,
                        first_frame_latents=current_clean,
                        mosaic_latents=mosaic,
                        mosaic_frame_indices=mosaic_indices,
                        drop_mosaic_holes=True,
                        sequence_parallel_size=1,
                    )
                    model_kwargs = {
                        "camera_info": camera_info,
                        "latent_layout": layout,
                        "height": int(config.matrixgame35_height),
                        "width": int(config.matrixgame35_width),
                    }
                    if batch.subject_ref_latents is not None:
                        model_kwargs["subject_ref_latents"] = batch.subject_ref_latents
                    with matrixgame35_autocast_context(device, dtype,
                                                       fastvideo_args.disable_autocast), set_forward_context(
                                                           current_timestep=step_index,
                                                           attn_metadata=None,
                                                           forward_batch=batch,
                                                       ):
                        positive_full = self.transformer(
                            layout.latents,
                            prompt,
                            layout.token_timesteps.reshape(1, -1),
                            **model_kwargs,
                        )
                        if do_classifier_free_guidance:
                            assert negative_prompt is not None
                            negative_full = self.transformer(
                                layout.latents,
                                negative_prompt,
                                layout.token_timesteps.reshape(1, -1),
                                **model_kwargs,
                            )
                    positive = positive_full[:, :, layout.output_frame_slice]
                    if positive.shape != noisy.shape:
                        raise ValueError(
                            "Matrix-Game 3.5 transformer output slice must match the 21 noisy latents, got "
                            f"{tuple(positive.shape)} and {tuple(noisy.shape)}.")
                    if do_classifier_free_guidance:
                        negative = negative_full[:, :, layout.output_frame_slice]
                        if negative.shape != noisy.shape:
                            raise ValueError(
                                "Matrix-Game 3.5 negative transformer output slice must match the 21 noisy latents, "
                                f"got {tuple(negative.shape)} and {tuple(noisy.shape)}.")
                        velocity = negative + batch.guidance_scale * (positive - negative)
                    else:
                        velocity = positive
                    next_sigma = schedule.sigmas[step_index + 1] if step_index + 1 < len(schedule.sigmas) else 0.0
                    noisy = base_flow_step(noisy, velocity, schedule.sigmas[step_index], next_sigma)
            finally:
                self._offload_transformer(fastvideo_args)

            generated = noisy
            generated_chunks.append(generated)
            section_decoded = self._decode_video(torch.cat((current_clean, generated), dim=2), fastvideo_args)
            if section_decoded.shape[2] != RGB_FRAMES_PER_BLOCK + 1:
                raise ValueError("one clean + 21 generated latents must decode to 85 RGB frames, got "
                                 f"{section_decoded.shape[2]}.")
            generated_decoded = section_decoded[:, :, 1:]
            current_clean = generated[:, :, -1:]

            if block_index + 1 < num_blocks:
                self._append_memory_frames(
                    memory,
                    generated_decoded,
                    w2c=memory_result.aligned_query_w2c,
                    intrinsics=target_intrinsics,
                    fastvideo_args=fastvideo_args,
                )
            del generated_decoded, section_decoded

        batch.latents = torch.cat((anchor_latent, *generated_chunks), dim=2)
        batch.raw_latent_shape = tuple(batch.latents.shape)
        batch.output = self._decode_video(batch.latents, fastvideo_args).detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if batch.output.shape[2] != batch.num_frames:
            raise RuntimeError(
                f"Matrix-Game 3.5 published {batch.output.shape[2]} frames; expected {batch.num_frames}.")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, lambda value: isinstance(value, torch.Tensor))
        result.add_check("prompt_embeds", batch.prompt_embeds,
                         lambda value: isinstance(value, list) and len(value) == 1)
        if float(batch.guidance_scale) != 1.0:
            result.add_check(
                "negative_prompt_embeds",
                batch.negative_prompt_embeds,
                lambda value: isinstance(value, list) and len(value) == 1,
            )
        result.add_check("depth_adapter", self.depth_adapter, lambda value: value is not None)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "output",
            batch.output,
            lambda value: isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[2] == batch.num_frames,
        )
        return result


__all__ = [
    "BASE_GUIDANCE_SCALE",
    "BASE_SEED",
    "LATENTS_PER_BLOCK",
    "MatrixGame35BaseInputValidationStage",
    "MatrixGame35BaseRolloutStage",
    "matrixgame35_base_block_seed",
    "preprocess_matrixgame35_anchor",
]
