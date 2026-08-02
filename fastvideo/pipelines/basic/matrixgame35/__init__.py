"""Matrix-Game 3.5 pipelines and shared components."""

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
from fastvideo.pipelines.basic.matrixgame35.layout import (
    MatrixGame35LatentLayout,
    build_noncausal_latent_layout,
)
from fastvideo.pipelines.basic.matrixgame35.schedule import (
    MatrixGame35DistilledSchedule,
    build_distilled_schedule,
    distilled_noise_seeds,
    x0_renoise_transition,
)
from fastvideo.pipelines.basic.matrixgame35.base_first_person_pipeline import MatrixGame35BaseFirstPersonPipeline
from fastvideo.pipelines.basic.matrixgame35.base_third_person_pipeline import MatrixGame35BaseThirdPersonPipeline
from fastvideo.pipelines.basic.matrixgame35.distilled_standard_pipeline import MatrixGame35DistilledFirstPersonPipeline

__all__ = [
    "MatrixGame35BaseFirstPersonPipeline",
    "MatrixGame35BaseThirdPersonPipeline",
    "MatrixGame35DistilledFirstPersonPipeline",
    "MatrixGame35CameraTrajectory",
    "build_prope_viewmats",
    "gather_latent_subframes",
    "load_camera_trajectory",
    "required_camera_frames",
    "build_mosaic_cross_attention_keep_mask",
    "build_subject_ref_memory_tokens",
    "prepend_subject_ref_prope_camera_info",
    "MatrixGame35LatentLayout",
    "build_noncausal_latent_layout",
    "MatrixGame35DistilledSchedule",
    "build_distilled_schedule",
    "distilled_noise_seeds",
    "x0_renoise_transition",
]
