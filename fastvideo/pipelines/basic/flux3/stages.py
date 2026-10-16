# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image

from fastvideo.distributed import get_local_torch_device
from fastvideo.logger import init_logger
from fastvideo.models.dits.flux3 import IN_CHANNELS
from fastvideo.models.encoders.flux3_pup import bbox as flux3_bbox
from fastvideo.models.vaes.flux3 import latent_ids, latent_shape
from fastvideo.pipelines.basic.flux3.image_processing import (
    MAX_REFERENCES,
    default_size,
    prepare_multi_reference,
    prepare_reference,
    to_tensor,
)
from fastvideo.pipelines.basic.flux3.packing import pack, scaled_ids
from fastvideo.pipelines.basic.flux3.sampling import denoise_flow_map, timesteps
from fastvideo.pipelines.basic.flux3.state import get_state
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


def _check(name: str, value: Any, predicate: Callable[[Any], bool]) -> VerificationResult:
    result = VerificationResult()
    result.add_check(name, value, predicate)
    return result


class Flux3InputPreparationStage(PipelineStage):

    def verify_input(self, batch, fastvideo_args):
        return _check("prompt", batch.prompt, lambda value: isinstance(value, str))

    def verify_output(self, batch, fastvideo_args):
        return _check("references", batch.references, lambda value: isinstance(value, list))

    def forward(self, batch, fastvideo_args):
        if not isinstance(batch.prompt, str):
            raise ValueError("FLUX.3 accepts one prompt per request")
        if batch.prompt_embeds:
            raise ValueError("FLUX.3 requires a text prompt; precomputed prompt_embeds are not supported")
        if batch.sigmas is not None:
            raise ValueError("FLUX.3 uses flux3_alpha and num_inference_steps; custom sigmas are not supported")
        if batch.negative_prompt:
            raise ValueError("FLUX.3 distilled does not use negative_prompt")
        text_len = fastvideo_args.pipeline_config.text_encoder_configs[0].text_len
        if batch.max_sequence_length not in (None, text_len):
            raise ValueError(f"FLUX.3 uses a fixed max_sequence_length of {text_len}")
        if batch.num_frames != 1 or batch.num_videos_per_prompt != 1:
            raise ValueError("FLUX.3 produces one image per request")
        if batch.guidance_scale != 1.0 or batch.true_cfg_scale != 1.0 or batch.use_embedded_guidance:
            raise ValueError("FLUX.3 distilled samples without CFG; use guidance_scale=1")
        if not isinstance(batch.num_inference_steps, int) or batch.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if not math.isfinite(batch.flux3_alpha) or batch.flux3_alpha <= 0:
            raise ValueError("flux3_alpha must be positive and finite")
        if not isinstance(batch.flux3_pup_max_new_tokens, int) or batch.flux3_pup_max_new_tokens <= 0:
            raise ValueError("flux3_pup_max_new_tokens must be a positive integer")
        if batch.return_trajectory_latents or batch.return_trajectory_decoded:
            raise ValueError("FLUX.3 does not support trajectory output")
        if sum(value is not None for value in (batch.references, batch.image_path, batch.pil_image)) > 1:
            raise ValueError("Use only one of references, image_path or pil_image")
        references = batch.references
        if references is None:
            reference = batch.image_path if batch.image_path is not None else batch.pil_image
            references = [] if reference is None else [reference]
        if not isinstance(references, list) or len(references) > MAX_REFERENCES:
            raise ValueError(f"FLUX.3 accepts a list of at most {MAX_REFERENCES} references")
        images = []
        for reference in references:
            if isinstance(reference, str | Path):
                with Image.open(reference) as image:
                    images.append(image.copy())
            elif isinstance(reference, Image.Image):
                images.append(reference.copy())
            else:
                raise TypeError("FLUX.3 references must be image paths or PIL images")
        if not batch.flux3_auto_size:
            if not isinstance(batch.height, int) or not isinstance(batch.width, int):
                raise ValueError("FLUX.3 height and width must be positive multiples of 16")
            if batch.height <= 0 or batch.width <= 0:
                raise ValueError("FLUX.3 height and width must be positive multiples of 16")
            latent_shape(batch.height, batch.width)
        if batch.flux3_boxes:
            flux3_bbox.to_canvas(batch.flux3_boxes)
        batch.references = images
        batch.batch_size = 1
        batch.do_classifier_free_guidance = False
        return batch


class Flux3PromptUpsamplingStage(PipelineStage):
    performance_component_metric = "text_encoder"

    def __init__(self, upsampler, processor, system_prompts):
        self.upsampler = upsampler
        self.processor = processor
        self.system_prompts = system_prompts

    def verify_input(self, batch, fastvideo_args):
        return _check("prompt", batch.prompt, lambda value: isinstance(value, str))

    def verify_output(self, batch, fastvideo_args):
        return _check("seed", batch.seed, lambda value: isinstance(value, int))

    def forward(self, batch, fastvideo_args):
        finished = not batch.flux3_boxes and flux3_bbox.is_finished(batch.prompt)
        caption = None
        if batch.flux3_upsample_prompt and not finished:
            self.upsampler.to(get_local_torch_device())
            try:
                caption = self.upsampler.upsample(batch.prompt,
                                                  batch.references,
                                                  batch.flux3_boxes,
                                                  max_new_tokens=batch.flux3_pup_max_new_tokens,
                                                  processor=self.processor,
                                                  system_prompts=self.system_prompts)
            except ValueError as error:
                logger.warning("FLUX.3 prompt upsampling skipped: %s", error)
            finally:
                if fastvideo_args.text_encoder_cpu_offload:
                    self.upsampler.to("cpu")
        if caption is None:
            caption = (
                f"{batch.prompt.rstrip()} {json.dumps(flux3_bbox.to_canvas(batch.flux3_boxes), ensure_ascii=False)}"
                if batch.flux3_boxes else batch.prompt)
        batch.prompt = caption
        if batch.seed is None:
            batch.seed = int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32
        batch.seeds = [batch.seed]
        batch.generator = torch.Generator("cpu").manual_seed(batch.seed)
        return batch


class Flux3TextEncodingStage(PipelineStage):
    performance_component_metric = "text_encoder"

    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def verify_input(self, batch, fastvideo_args):
        return _check("prompt", batch.prompt, lambda value: isinstance(value, str))

    def verify_output(self, batch, fastvideo_args):
        return _check("prompt_embeds", batch.prompt_embeds, lambda value: len(value) == 1 and value[0].ndim == 2)

    def forward(self, batch, fastvideo_args):
        self.text_encoder.to(get_local_torch_device())
        try:
            tokens, ids = self.text_encoder.encode(batch.prompt, self.tokenizer)
            batch.prompt_embeds = [tokens]
            get_state(batch).text_ids = ids
        finally:
            if fastvideo_args.text_encoder_cpu_offload:
                self.text_encoder.to("cpu")
        return batch


class Flux3LatentPreparationStage(PipelineStage):
    performance_component_metric = "vae"

    def __init__(self, vae):
        self.vae = vae

    def verify_input(self, batch, fastvideo_args):
        return _check("prompt_embeds", batch.prompt_embeds, lambda value: len(value) == 1)

    def verify_output(self, batch, fastvideo_args):
        return _check("packed", get_state(batch).packed, lambda value: value is not None)

    def forward(self, batch, fastvideo_args):
        references = batch.references
        state = get_state(batch)
        state.crop = None
        if len(references) > 1:
            references = [prepare_multi_reference(image) for image in references]
        else:
            prepared = [prepare_reference(image) for image in references]
            references = [image for image, _ in prepared]
            if prepared and batch.flux3_auto_size:
                state.crop = prepared[0][1]
        if batch.flux3_auto_size:
            batch.width, batch.height = default_size(references)
        h, w = latent_shape(batch.height, batch.width)
        device = get_local_torch_device()
        condition = None
        if references:
            self.vae.to(device)
            tokens, ids = [], []
            try:
                for index, reference in enumerate(references):
                    if min(reference.size) == 0:
                        raise ValueError("FLUX.3 multi-reference images must have sides of at least 16 pixels")
                    t, i = self.vae.encode(to_tensor(reference).to(device))
                    i = scaled_ids(i)
                    i[:, 3] = -(1 + index)
                    tokens.append(t.to(torch.bfloat16))
                    ids.append(i)
                condition = torch.cat(tokens), torch.cat(ids)
            finally:
                if fastvideo_args.vae_cpu_offload:
                    self.vae.to("cpu")
        if batch.latents is None:
            noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=batch.generator)
            noise = noise.to(torch.bfloat16).float().to(device)
        else:
            if tuple(batch.latents.shape) != (h * w, IN_CHANNELS["image"]):
                raise ValueError(f"FLUX.3 latents must have shape {(h * w, IN_CHANNELS['image'])}")
            noise = batch.latents.to(device=device, dtype=torch.float32)
        state.image_ids = latent_ids(h, w, device)
        state.packed = pack([[
            (batch.prompt_embeds[0], state.text_ids),
            condition,
            (noise, scaled_ids(state.image_ids)),
        ]])
        batch.timesteps = timesteps(batch.num_inference_steps, batch.flux3_alpha)
        batch.latents = noise
        return batch


class Flux3DenoisingStage(PipelineStage):
    performance_component_metric = "dit"

    def __init__(self, transformer):
        self.transformer = transformer

    def verify_input(self, batch, fastvideo_args):
        return _check("packed", get_state(batch).packed, lambda value: value is not None)

    def verify_output(self, batch, fastvideo_args):
        return _check("latents", batch.latents, lambda value: isinstance(value, torch.Tensor) and value.ndim == 2)

    def forward(self, batch, fastvideo_args):
        self.transformer.to(get_local_torch_device())
        if not self.transformer.flow_map:
            raise ValueError("FLUX.3 requires the distilled flow-map checkpoint")
        state = get_state(batch)
        batch.latents = denoise_flow_map(self.transformer, state.packed, batch.timesteps)
        state.packed = None
        return batch


class Flux3DecodingStage(PipelineStage):
    performance_component_metric = "vae"

    def __init__(self, vae):
        self.vae = vae

    def verify_input(self, batch, fastvideo_args):
        return _check("latents", batch.latents, lambda value: isinstance(value, torch.Tensor))

    def verify_output(self, batch, fastvideo_args):
        return _check("output", batch.output, lambda value: isinstance(value, torch.Tensor))

    def forward(self, batch, fastvideo_args):
        if fastvideo_args.output_type == "latent":
            batch.output = batch.latents
            return batch
        self.vae.to(get_local_torch_device())
        try:
            state = get_state(batch)
            image = self.vae.decode(batch.latents, state.image_ids)
            if state.crop is not None:
                left, top, right, bottom = state.crop
                image = image[:, top:bottom, left:right]
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            batch.output = (pixels.float() / 255)[None, :, None]
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")
        return batch
