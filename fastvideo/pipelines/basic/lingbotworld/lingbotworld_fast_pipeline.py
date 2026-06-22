# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-Fast causal DMD image-to-video pipeline.

Single block-causal DMD transformer, no CLIP image encoder. The first frame is
conditioned via Wan-2.1 channel concatenation (built by the image VAE encoding
stage and concatenated inside the causal denoising stage).
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, InputValidationStage, LatentPreparationStage,
                                        TextEncodingStage)
from fastvideo.pipelines.stages.image_encoding import (MatrixGame2ImageVAEEncodingStage)
from fastvideo.pipelines.stages.lingbotworld_fast_denoising import (LingBotWorldFastCausalDenoisingStage)

logger = init_logger(__name__)


class LingBotWorldCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["vae", "transformer", "scheduler"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        if (self.get_module("text_encoder", None) is not None and self.get_module("tokenizer", None) is not None):
            self.add_stage(stage_name="prompt_encoding_stage",
                           stage=TextEncodingStage(
                               text_encoders=[self.get_module("text_encoder")],
                               tokenizers=[self.get_module("tokenizer")],
                           ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=MatrixGame2ImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=LingBotWorldFastCausalDenoisingStage(transformer=self.get_module("transformer"),
                                                                  scheduler=self.get_module("scheduler"),
                                                                  pipeline=self,
                                                                  vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))

        logger.info("LingBotWorldCausalDMDPipeline initialized")


EntryClass = LingBotWorldCausalDMDPipeline
