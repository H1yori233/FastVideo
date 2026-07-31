"""Matrix-Game 3.5 pipeline components.

The production pipeline is activated only after component parity is complete.
"""

from fastvideo.pipelines.basic.matrixgame35.camera import (
    MatrixGame35CameraTrajectory,
    build_prope_viewmats,
    gather_latent_subframes,
    load_camera_trajectory,
    required_camera_frames,
)
from fastvideo.pipelines.basic.matrixgame35.conditioning import (
    build_mosaic_cross_attention_keep_mask,
    build_subject_ref_memory_tokens,
    prepend_subject_ref_prope_camera_info,
)
from fastvideo.pipelines.basic.matrixgame35.schedule import (
    MatrixGame35DistilledSchedule,
    build_distilled_schedule,
    distilled_noise_seeds,
    x0_renoise_transition,
)

__all__ = [
    "MatrixGame35CameraTrajectory",
    "build_prope_viewmats",
    "gather_latent_subframes",
    "load_camera_trajectory",
    "required_camera_frames",
    "build_mosaic_cross_attention_keep_mask",
    "build_subject_ref_memory_tokens",
    "prepend_subject_ref_prope_camera_info",
    "MatrixGame35DistilledSchedule",
    "build_distilled_schedule",
    "distilled_noise_seeds",
    "x0_renoise_transition",
]
