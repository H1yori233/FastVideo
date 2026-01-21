# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.matrixgame.matrixgame_causal_dmd_pipeline import (
    MatrixGameCausalDMDPipeline)
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.self_forcing_distillation_pipeline import (
    SelfForcingDistillationPipeline)
from fastvideo.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class MatrixGameSelfForcingDistillationPipeline(SelfForcingDistillationPipeline
                                                ):
    """
    A self-forcing distillation pipeline for MatrixGame that uses the self-forcing methodology
    with DMD for video generation.
    """
    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
    ]

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        validation_pipeline = MatrixGameCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "transformer_2": self.get_module("transformer_2")
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        clip_feature = batch['clip_feature']
        first_frame_latent = batch['first_frame_latent']
        keyboard_cond = batch.get('keyboard_cond', None)
        keyboard_cond = keyboard_cond[:, :3]  # TODO: remove hardcode
        mouse_cond = batch.get('mouse_cond', None)
        infos = batch['info_list']

        batch_size = clip_feature.shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.z_dim
        spatial_compression_ratio = vae_config.spatial_compression_ratio

        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio

        latents = torch.randn(batch_size, num_channels,
                              self.training_args.num_latent_t, latent_height,
                              latent_width).to(get_local_torch_device(),
                                               dtype=torch.bfloat16)

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.image_embeds = clip_feature.to(get_local_torch_device(),
                                                      dtype=torch.bfloat16)
        training_batch.image_latents = first_frame_latent.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        # Action conditioning
        if keyboard_cond is not None and keyboard_cond.numel() > 0:
            training_batch.keyboard_cond = keyboard_cond.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.keyboard_cond = None
        if mouse_cond is not None and mouse_cond.numel() > 0:
            training_batch.mouse_cond = mouse_cond.to(get_local_torch_device(),
                                                      dtype=torch.bfloat16)
        else:
            training_batch.mouse_cond = None
        training_batch.infos = infos
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""

        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        temporal_compression_ratio = 4
        num_frames = (self.training_args.num_latent_t -
                      1) * temporal_compression_ratio + 1
        batch_size, num_channels, _, latent_height, latent_width = image_latents.shape
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, 1:] = 0

        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=temporal_compression_ratio)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]],
                                  dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1,
                                           temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(
            image_latents.device).to(dtype=torch.bfloat16)

        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, mask_lat_size, image_latents],
            dim=1)

        return training_batch


def main(args) -> None:
    logger.info("Starting MatrixGame self-forcing distillation pipeline...")

    pipeline = MatrixGameSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    pipeline.train()
    logger.info("MatrixGame self-forcing distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
