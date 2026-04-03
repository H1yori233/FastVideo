# SPDX-License-Identifier: Apache-2.0
import sys
import os
from copy import deepcopy
from typing import Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_matrixgame
from fastvideo.dataset.validation_dataset import ValidationDataset
from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_video
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.matrixgame.matrixgame_i2v_pipeline import (
    MatrixGamePipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.ptlflow_validation import (
    PTLFLOW_SCALAR_KEYS,
    PTLFlowValidationHelper,
)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.utils import is_vsa_available, shallow_asdict

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class MatrixGameTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Matrix-Game-2.0.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_matrixgame

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.dit_cpu_offload = True
        # args_copy.pipeline_config.vae_config.load_encoder = False
        # validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
        self.validation_pipeline = MatrixGamePipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            dit_cpu_offload=True)
        self._ptlflow_validation = PTLFlowValidationHelper()

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        # encoder_hidden_states = batch['text_embedding']
        # encoder_attention_mask = batch['text_attention_mask']
        clip_features = batch['clip_feature']
        image_latents = batch['first_frame_latent']
        image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = batch['pil_image']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        # MatrixGame doesn't use text encoder
        training_batch.preprocessed_image = pil_image.to(
            get_local_torch_device())
        training_batch.image_embeds = clip_features.to(get_local_torch_device())
        training_batch.image_latents = image_latents.to(
            get_local_torch_device())
        training_batch.infos = infos

        # Action conditioning
        if 'mouse_cond' in batch and batch['mouse_cond'].numel() > 0:
            training_batch.mouse_cond = batch['mouse_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.mouse_cond = None

        if 'keyboard_cond' in batch and batch['keyboard_cond'].numel() > 0:
            training_batch.keyboard_cond = batch['keyboard_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.keyboard_cond = None

        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""

        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        temporal_compression_ratio = self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
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

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:

        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)
        encoder_hidden_states_image = image_embeds

        # NOTE: noisy_model_input already contains concatenated image_latents from _prepare_dit_inputs
        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,  # None for MatrixGame
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            # "encoder_attention_mask":
            # training_batch.encoder_attention_mask,
            "encoder_hidden_states_image":
            encoder_hidden_states_image,
            # Action conditioning
            "mouse_cond":
            training_batch.mouse_cond,
            "keyboard_cond":
            training_batch.keyboard_cond,
            "return_dict":
            False,
        }
        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch.get(
            'image_path') or validation_batch.get('video_path')
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )
        if "image" in validation_batch and validation_batch["image"] is not None:
            batch.pil_image = validation_batch["image"]

        if "keyboard_cond" in validation_batch and validation_batch[
                "keyboard_cond"] is not None:
            keyboard_cond = validation_batch["keyboard_cond"]
            keyboard_cond = torch.tensor(
                keyboard_cond[:sampling_param.num_frames],
                dtype=torch.bfloat16,
            )
            keyboard_cond = keyboard_cond.unsqueeze(0)
            batch.keyboard_cond = keyboard_cond

        if "mouse_cond" in validation_batch and validation_batch[
                "mouse_cond"] is not None:
            mouse_cond = validation_batch["mouse_cond"]
            mouse_cond = torch.tensor(
                mouse_cond[:sampling_param.num_frames],
                dtype=torch.bfloat16,
            )
            mouse_cond = mouse_cond.unsqueeze(0)
            batch.mouse_cond = mouse_cond

        return batch

    def _post_process_validation_frames(
            self, frames: list[np.ndarray],
            batch: ForwardBatch) -> list[np.ndarray]:
        from fastvideo.models.dits.matrixgame.utils import (
            overlay_validation_actions_on_frames,
        )

        return overlay_validation_actions_on_frames(
            frames,
            keyboard_cond=getattr(batch, "keyboard_cond", None),
            mouse_cond=getattr(batch, "mouse_cond", None),
        )

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        training_args.inference_mode = True
        training_args.dit_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        logger.info(
            "rank: %s: fastvideo_args.validation_dataset_file: %s",
            self.global_rank,
            training_args.validation_dataset_file,
            local_main_process_only=False,
        )
        local_validation_ok = 1
        local_validation_error = None
        try:
            validation_dataset = ValidationDataset(
                training_args.validation_dataset_file
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=None,
                num_workers=0,
            )
        except Exception as exc:
            local_validation_ok = 0
            local_validation_error = repr(exc)
            logger.warning(
                "Rank %s failed to build validation dataset, will skip validation this round. err=%s",
                self.global_rank,
                local_validation_error,
                local_main_process_only=False,
            )

        validation_ok = torch.tensor(
            local_validation_ok,
            device=get_local_torch_device(),
            dtype=torch.int32,
        )
        dist.all_reduce(validation_ok, op=dist.ReduceOp.MIN)
        if validation_ok.item() == 0:
            if self.global_rank == 0:
                logger.warning(
                    "Skip validation at step %s because at least one rank failed to prepare validation dataset.",
                    global_step,
                )
            training_args.inference_mode = False
            self.transformer.train()
            if getattr(self, "transformer_2", None) is not None:
                self.transformer_2.train()
            return

        self.transformer.eval()
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]
        world_group = get_world_group()
        num_sp_groups = world_group.world_size // self.sp_group.world_size
        evaluate_ptlflow = True

        for num_inference_steps in validation_steps:
            logger.info(
                "rank: %s: num_inference_steps: %s",
                self.global_rank,
                num_inference_steps,
                local_main_process_only=False,
            )
            step_videos: list[np.ndarray] = []
            step_captions: list[str] = []
            step_ref_videos: list[str | None] = []
            step_action_paths: list[str | None] = []
            step_audio: list[np.ndarray | None] = []
            step_sample_rates: list[int | None] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_batch(
                    sampling_param,
                    training_args,
                    validation_batch,
                    num_inference_steps,
                )
                action_path = validation_batch.get("action_path")
                if not isinstance(action_path, str):
                    action_path = None

                logger.info(
                    "rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                    self.global_rank,
                    self.rank_in_sp_group,
                    batch.prompt,
                    local_main_process_only=False,
                )

                assert batch.prompt is not None and isinstance(batch.prompt, str)
                output_batch = self.validation_pipeline.forward(batch, training_args)
                samples = output_batch.output.cpu()
                audio = output_batch.extra.get("audio")
                sample_rate = output_batch.extra.get("audio_sample_rate")

                if audio is not None and torch.is_tensor(audio):
                    audio = audio.detach().cpu().float().numpy()

                step_audio.append(audio)
                step_sample_rates.append(sample_rate)

                if self.rank_in_sp_group != 0:
                    continue

                step_captions.append(batch.prompt)
                step_ref_videos.append(validation_batch.get("ref_video"))
                step_action_paths.append(action_path)

                video = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                frames = self._post_process_validation_frames(frames, batch)
                step_videos.append(frames)

            if self.rank_in_sp_group == 0 and self.global_rank == 0:
                all_videos = list(step_videos)
                all_captions = list(step_captions)
                all_ref_videos = list(step_ref_videos)
                all_action_paths = list(step_action_paths)
                all_audios = list(step_audio)
                all_sample_rates = list(step_sample_rates)

                for sp_group_idx in range(1, num_sp_groups):
                    src_rank = sp_group_idx * self.sp_world_size
                    recv_videos = world_group.recv_object(src=src_rank)
                    recv_captions = world_group.recv_object(src=src_rank)
                    recv_ref_videos = world_group.recv_object(src=src_rank)
                    recv_action_paths = world_group.recv_object(src=src_rank)
                    recv_audios = world_group.recv_object(src=src_rank)
                    recv_sample_rates = world_group.recv_object(src=src_rank)

                    all_videos.extend(recv_videos)
                    all_captions.extend(recv_captions)
                    all_ref_videos.extend(recv_ref_videos)
                    all_action_paths.extend(recv_action_paths)
                    all_audios.extend(recv_audios)
                    all_sample_rates.extend(recv_sample_rates)

                os.makedirs(training_args.output_dir, exist_ok=True)
                video_filenames: list[str] = []
                for i, (video, audio, sample_rate) in enumerate(
                    zip(
                        all_videos,
                        all_audios,
                        all_sample_rates,
                        strict=True,
                    )
                ):
                    filename = os.path.join(
                        training_args.output_dir,
                        f"validation_step_{global_step}_inference_steps_{num_inference_steps}"
                        f"_video_{i}.mp4",
                    )
                    imageio.mimsave(filename, video, fps=sampling_param.fps)
                    if (
                        audio is not None
                        and sample_rate is not None
                        and not self._mux_audio(filename, audio, sample_rate)
                    ):
                        logger.warning(
                            "Audio mux failed for validation video %s; saved video without audio.",
                            filename,
                        )
                    video_filenames.append(filename)

                artifacts = []
                for filename, caption in zip(
                    video_filenames,
                    all_captions,
                    strict=True,
                ):
                    video_artifact = self.tracker.video(filename, caption=caption)
                    if video_artifact is not None:
                        artifacts.append(video_artifact)
                if artifacts:
                    self.tracker.log_artifacts(
                        {
                            f"validation_videos_{num_inference_steps}_steps": artifacts
                        },
                        global_step,
                    )
                if not self.validation_ref_videos_logged:
                    ref_artifacts = []
                    for filename, caption in zip(
                        all_ref_videos,
                        all_captions,
                        strict=True,
                    ):
                        if filename is None:
                            continue
                        ref_frames = np.stack(
                            [np.asarray(frame) for frame in load_video(filename)],
                            axis=0,
                        )
                        ref_frames = np.ascontiguousarray(
                            ref_frames.transpose(0, 3, 1, 2)
                        )
                        video_artifact = self.tracker.video(
                            ref_frames,
                            caption=caption,
                            fps=sampling_param.fps,
                        )
                        if video_artifact is not None:
                            ref_artifacts.append(video_artifact)
                    if ref_artifacts:
                        self.tracker.log_artifacts(
                            {"validation_ref_videos": ref_artifacts},
                            global_step,
                        )
                        self.validation_ref_videos_logged = True

                if evaluate_ptlflow:
                    self._ptlflow_validation.initialize(training_args)
                    if not self._ptlflow_validation.ready:
                        logger.warning(
                            "PTLFlow evaluation is enabled but evaluator initialization failed. "
                            "Skipping flow metrics for this validation run."
                        )
                    else:
                        metric_sums = {
                            key: 0.0 for key in PTLFLOW_SCALAR_KEYS
                        }
                        metric_counts = {
                            key: 0.0 for key in PTLFLOW_SCALAR_KEYS
                        }
                        for filename, action_path in zip(
                            video_filenames,
                            all_action_paths,
                            strict=True,
                        ):
                            try:
                                sample_metrics = self._ptlflow_validation.evaluate_video(
                                    video_path=filename,
                                    action_path=action_path,
                                    global_step=global_step,
                                    num_inference_steps=num_inference_steps,
                                    training_args=training_args,
                                )
                                for key in PTLFLOW_SCALAR_KEYS:
                                    val = sample_metrics.get(key)
                                    if not isinstance(
                                        val,
                                        (float, int, np.floating, np.integer),
                                    ):
                                        continue
                                    val_float = float(val)
                                    if not np.isfinite(val_float):
                                        continue
                                    metric_sums[key] += val_float
                                    metric_counts[key] += 1.0
                            finally:
                                self._ptlflow_validation.release_cuda_memory()

                        metric_logs: dict[str, float] = {}
                        for metric_key in PTLFLOW_SCALAR_KEYS:
                            count = metric_counts[metric_key]
                            if count <= 0:
                                continue
                            value = float(metric_sums[metric_key] / count)
                            if not np.isfinite(value):
                                continue
                            metric_logs[f"metrics/{metric_key}"] = value
                        if metric_logs:
                            self.tracker.log(metric_logs, global_step)

            elif self.rank_in_sp_group == 0:
                world_group.send_object(step_videos, dst=0)
                world_group.send_object(step_captions, dst=0)
                world_group.send_object(step_ref_videos, dst=0)
                world_group.send_object(step_action_paths, dst=0)
                world_group.send_object(step_audio, dst=0)
                world_group.send_object(step_sample_rates, dst=0)

        training_args.inference_mode = False
        self.transformer.train()
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.train()


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = MatrixGameTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
