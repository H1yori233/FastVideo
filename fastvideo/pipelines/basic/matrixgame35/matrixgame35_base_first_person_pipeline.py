# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 Base first-person STANDARD pipeline composition."""

from typing import Any

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.matrixgame35 import MatrixGame35BaseFirstPersonPipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.matrixgame35.base_stages import (
    MatrixGame35BaseInputValidationStage,
    MatrixGame35BaseRolloutStage,
)
from fastvideo.pipelines.basic.matrixgame35.depth_estimation import MatrixGame35DepthAnything3Adapter
from fastvideo.pipelines.basic.matrixgame35.patch_memory import MatrixGame35DepthAdapter
from fastvideo.pipelines.basic.matrixgame35.prompts import MatrixGame35TextEncodingStage
from fastvideo.pipelines.stages import ConditioningStage


class MatrixGame35BaseFirstPersonPipeline(LoRAPipeline, ComposedPipelineBase):
    """Released Base first-person STANDARD pipeline."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer"]
    pipeline_config_cls = MatrixGame35BaseFirstPersonPipelineConfig
    sampling_params_cls = SamplingParam
    input_validation_stage_cls = MatrixGame35BaseInputValidationStage

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        self._depth_adapter: MatrixGame35DepthAdapter | None = MatrixGame35DepthAnything3Adapter(
            model_ref=fastvideo_args.pipeline_config.matrixgame35_da3_model_ref,
            device=get_local_torch_device(),
            cpu_offload=True,
        )

    def _add_variant_conditioning_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Extension point for Base variants that add conditioning before rollout."""

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage("input_validation_stage", self.input_validation_stage_cls())
        self.add_stage(
            "prompt_encoding_stage",
            MatrixGame35TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage("conditioning_stage", ConditioningStage())
        self._add_variant_conditioning_stages(fastvideo_args)
        self.add_stage(
            "base_rollout_stage",
            MatrixGame35BaseRolloutStage(
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
                depth_adapter=self._depth_adapter,
            ),
        )

    def set_depth_adapter(self, depth_adapter: MatrixGame35DepthAdapter | None) -> None:
        """Inject a DA3-compatible adapter without importing an optional backend eagerly."""
        self._depth_adapter = depth_adapter
        stage: Any = self._stage_name_mapping.get("base_rollout_stage")
        if stage is not None:
            stage.set_depth_adapter(depth_adapter)


EntryClass = MatrixGame35BaseFirstPersonPipeline

__all__ = ["MatrixGame35BaseFirstPersonPipeline"]
