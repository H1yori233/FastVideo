# SPDX-License-Identifier: Apache-2.0
"""Public re-exports for Matrix-Game 3.5 conditioning helpers."""

from fastvideo.models.dits._matrixgame35_conditioning import (
    build_mosaic_cross_attention_keep_mask,
    build_subject_ref_memory_tokens,
    prepend_subject_ref_prope_camera_info,
)

__all__ = [
    "build_mosaic_cross_attention_keep_mask",
    "build_subject_ref_memory_tokens",
    "prepend_subject_ref_prope_camera_info",
]
