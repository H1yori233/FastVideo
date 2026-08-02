# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 distilled first-person runtime-profile composition."""

from typing import Any

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.matrixgame35 import MatrixGame35DistilledFirstPersonPipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.matrixgame35.distilled_memory import (
    MatrixGame35DistilledDepthAnything3Adapter, )
from fastvideo.pipelines.basic.matrixgame35.distilled_stages import (
    MatrixGame35DistilledInputValidationStage,
    MatrixGame35DistilledRolloutStage,
)
from fastvideo.pipelines.basic.matrixgame35.patch_memory import MatrixGame35DepthAdapter
from fastvideo.pipelines.basic.matrixgame35.prompts import MatrixGame35TextEncodingStage
from fastvideo.pipelines.stages import ConditioningStage


class MatrixGame35DistilledFirstPersonPipeline(LoRAPipeline, ComposedPipelineBase):
    """Released Distilled first-person pipeline with selectable runtime profiles."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer"]
    pipeline_config_cls = MatrixGame35DistilledFirstPersonPipelineConfig
    sampling_params_cls = SamplingParam

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        model_ref = fastvideo_args.pipeline_config.matrixgame35_da3_model_ref
        self._depth_adapter: MatrixGame35DepthAdapter | None = MatrixGame35DistilledDepthAnything3Adapter(
            model_ref,
            device=get_local_torch_device(),
            cpu_offload=True,
        )

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage("input_validation_stage", MatrixGame35DistilledInputValidationStage())
        self.add_stage(
            "prompt_encoding_stage",
            MatrixGame35TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage("conditioning_stage", ConditioningStage())
        self.add_stage(
            "distilled_rollout_stage",
            MatrixGame35DistilledRolloutStage(
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
                depth_adapter=self._depth_adapter,
            ),
        )

    def set_depth_adapter(self, depth_adapter: MatrixGame35DepthAdapter | None) -> None:
        """Inject a pinned DA3-compatible adapter without eager optional imports."""
        self._depth_adapter = depth_adapter
        stage: Any = self._stage_name_mapping.get("distilled_rollout_stage")
        if stage is not None:
            stage.set_depth_adapter(depth_adapter)


EntryClass = MatrixGame35DistilledFirstPersonPipeline

__all__ = ["MatrixGame35DistilledFirstPersonPipeline"]
