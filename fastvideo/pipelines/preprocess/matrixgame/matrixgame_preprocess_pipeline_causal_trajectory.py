# SPDX-License-Identifier: Apache-2.0
"""
Causal MatrixGame Trajectory Data Preprocessing pipeline implementation.

This module contains an implementation of the Causal MatrixGame Trajectory Data Preprocessing pipeline.
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.pipelines.preprocess.matrixgame.matrixgame_preprocess_pipeline_ode_trajectory import (
    PreprocessPipeline_MatrixGame_ODE_Trajectory)
from fastvideo.pipelines.stages import (DecodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        MatrixGameCausalDenoisingStage,
                                        MatrixGameImageEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)


class PreprocessPipeline_MatrixGame_Causal_Trajectory(PreprocessPipeline_MatrixGame_ODE_Trajectory):
    """Causal MatrixGame Trajectory preprocessing pipeline implementation."""

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        assert fastvideo_args.pipeline_config.flow_shift == 5
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)

        num_inference_steps = getattr(fastvideo_args.pipeline_config, "num_inference_steps", 3)
        self.modules["scheduler"].set_timesteps(num_inference_steps=num_inference_steps,
                                                denoising_strength=1.0)

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())
        
        self.add_stage(stage_name="image_encoding_stage",
                       stage=MatrixGameImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("image_processor"),
                       ))
        
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))
        
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))
        
        # uce matrixgame causal denoising
        self.add_stage(stage_name="denoising_stage",
                       stage=MatrixGameCausalDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler"),
                           pipeline=self,
                           vae=self.get_module("vae")
                       ))
        
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def preprocess_action_and_trajectory(self, fastvideo_args: FastVideoArgs, args):
        logger.info("Running Causal MatrixGame preprocessing...")
        return super().preprocess_action_and_trajectory(fastvideo_args, args)


EntryClass = PreprocessPipeline_MatrixGame_Causal_Trajectory
