# SPDX-License-Identifier: Apache-2.0
"""Third-person subject-reference stages for Matrix-Game 3.5 Base."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.matrixgame35.base_stages import (
    MatrixGame35BaseInputValidationStage, )
from fastvideo.pipelines.basic.matrixgame35.codec import encode_matrixgame35_video
from fastvideo.pipelines.basic.matrixgame35.runtime import run_matrixgame35_vae_operation
from fastvideo.pipelines.basic.matrixgame35.subject_references import load_subject_reference_canvases
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage


class MatrixGame35BaseThirdPersonInputValidationStage(MatrixGame35BaseInputValidationStage):
    """Apply Base validation while permitting third-person subject references."""

    allow_subject_refs = True


class MatrixGame35BaseThirdPersonSubjectReferenceStage(PipelineStage):
    """Load and independently encode the optional third-person subject references."""

    def __init__(self, vae: Any) -> None:
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        config = fastvideo_args.pipeline_config
        arch_config = config.dit_config.arch_config
        max_refs = int(arch_config.subject_ref_memory_max_refs)
        if max_refs != 4:
            raise ValueError("Matrix-Game 3.5 Base third-person requires subject_ref_memory_max_refs=4.")

        height = int(config.matrixgame35_height)
        width = int(config.matrixgame35_width)
        expected_shape = (int(arch_config.in_channels), 1, height // 16, width // 16)
        if batch.subject_ref_latents is not None:
            if batch.subject_ref_source is not None:
                raise ValueError("Provide either subject_ref_source or subject_ref_latents, not both.")
            latents = batch.subject_ref_latents
            if (not torch.is_tensor(latents) or latents.ndim != 5 or not 1 <= latents.shape[0] <= max_refs
                    or tuple(latents.shape[1:]) != expected_shape):
                shape = tuple(latents.shape) if torch.is_tensor(latents) else type(latents).__name__
                raise ValueError("Direct subject_ref_latents must have shape "
                                 f"[1..{max_refs},{expected_shape[0]},1,{expected_shape[2]},{expected_shape[3]}], "
                                 f"got {shape}.")
            batch.subject_ref_latents = latents.contiguous()
            return batch
        if batch.subject_ref_source is None:
            return batch

        canvases = load_subject_reference_canvases(
            batch.subject_ref_source,
            height=height,
            width=width,
            max_refs=max_refs,
        )
        if not torch.is_tensor(canvases) or canvases.ndim != 4:
            raise ValueError("Subject-reference loading must produce a tensor shaped [R,3,H,W].")
        if canvases.shape[0] > max_refs:
            raise ValueError(
                f"Base third-person accepts at most {max_refs} subject references, got {canvases.shape[0]}.")
        if canvases.shape[0] == 0:
            return batch
        if tuple(canvases.shape[1:]) != (3, height, width):
            raise ValueError("Subject-reference loading must produce "
                             f"[R,3,{height},{width}], got {tuple(canvases.shape)}.")

        expected_encode_shape = (1, *expected_shape)
        observed_shapes: list[tuple[int, ...]] = []

        def _encode_references(vae: Any, references: torch.Tensor) -> torch.Tensor:
            latents = []
            for canvas in references:
                latent = encode_matrixgame35_video(
                    vae,
                    canvas.unsqueeze(0).unsqueeze(2),
                )
                observed_shapes.append(tuple(latent.shape))
                if tuple(latent.shape) != expected_encode_shape:
                    raise ValueError("Each subject reference must encode to "
                                     f"{expected_encode_shape}, got {tuple(latent.shape)}.")
                latents.append(latent)
            return torch.cat(latents, dim=0).contiguous()

        batch.subject_ref_latents = run_matrixgame35_vae_operation(
            self.vae,
            canvases,
            precision=config.vae_precision,
            device=get_local_torch_device(),
            fastvideo_args=fastvideo_args,
            operation=_encode_references,
        )
        if len(observed_shapes) != int(canvases.shape[0]):
            raise RuntimeError("Each subject reference must be VAE-encoded exactly once.")
        return batch


__all__ = [
    "MatrixGame35BaseThirdPersonInputValidationStage",
    "MatrixGame35BaseThirdPersonSubjectReferenceStage",
]
