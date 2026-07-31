# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 Base third-person STANDARD pipeline composition."""

from fastvideo.configs.pipelines.matrixgame35 import MatrixGame35BaseThirdPersonPipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.matrixgame35.base_first_person_pipeline import (
    MatrixGame35BaseFirstPersonPipeline, )
from fastvideo.pipelines.basic.matrixgame35.base_third_person_stages import (
    MatrixGame35BaseThirdPersonInputValidationStage,
    MatrixGame35BaseThirdPersonSubjectReferenceStage,
)


class MatrixGame35BaseThirdPersonPipeline(MatrixGame35BaseFirstPersonPipeline):
    """Released Base third-person STANDARD pipeline, without public registry activation."""

    pipeline_config_cls = MatrixGame35BaseThirdPersonPipelineConfig
    input_validation_stage_cls = MatrixGame35BaseThirdPersonInputValidationStage

    def _add_variant_conditioning_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(
            "subject_reference_stage",
            MatrixGame35BaseThirdPersonSubjectReferenceStage(vae=self.get_module("vae")),
        )


EntryClass = MatrixGame35BaseThirdPersonPipeline

__all__ = ["MatrixGame35BaseThirdPersonPipeline"]
