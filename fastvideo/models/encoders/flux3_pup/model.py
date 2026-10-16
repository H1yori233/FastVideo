# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modified for FastVideo native multimodal generation.
"""Native multimodal PUP model and cached greedy decoding."""
from dataclasses import dataclass
from itertools import groupby

import torch
from torch import nn

from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.encoders.flux3_pup.policy import MAX_NEW_TOKENS, PRESENCE_PENALTY, STOP_TOKENS
from fastvideo.models.encoders.flux3_pup.text import Flux3PupTextModel
from fastvideo.models.encoders.flux3_pup.vision import Flux3PupVisionModel
from fastvideo.models.loader.weight_utils import default_weight_loader


@dataclass
class PromptUpsamplerOutput:
    logits: torch.Tensor
    past_key_values: object | None = None
    rope_deltas: torch.Tensor | None = None


class Flux3PupModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Flux3PupVisionModel(config.vision_config)
        self.language_model = Flux3PupTextModel(config.text_config)

    def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None, attention_mask=None):
        positions = torch.zeros(3, *input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)
        grids = iter(image_grid_thw) if image_grid_thw is not None else iter(())
        merge = self.config.vision_config.spatial_merge_size
        deltas = []
        for batch_idx, ids in enumerate(input_ids):
            types = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                ids = ids[attention_mask[batch_idx].bool()]
                types = types[attention_mask[batch_idx].bool()]
            current_pos, chunks = 0, []
            for modality, group in groupby(enumerate(types.tolist()), lambda item: item[1]):
                group = list(group)
                if modality == 0:
                    length = len(group)
                    chunks.append(torch.arange(length, device=ids.device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += length
                elif modality == 1:
                    grid = next(grids)
                    t, h, w = int(grid[0]), int(grid[1]) // merge, int(grid[2]) // merge
                    temporal = torch.arange(t, device=ids.device).repeat_interleave(h * w) + current_pos
                    height = (torch.arange(h, device=ids.device) + current_pos).repeat_interleave(w).repeat(t)
                    width = (torch.arange(w, device=ids.device) + current_pos).repeat(h * t)
                    chunks.append(torch.stack((temporal, height, width)))
                    current_pos += max(h, w)
                else:
                    raise ValueError("FLUX 3 prompt upsampling accepts images and text")
            current = torch.cat(chunks, dim=1)
            if attention_mask is not None:
                positions[:, batch_idx, attention_mask[batch_idx].bool()] = current
            else:
                positions[:, batch_idx] = current
            deltas.append(current.max() + 1 - len(ids))
        return positions, torch.stack(deltas).unsqueeze(1)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None,
                past_key_values=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None,
                rope_deltas=None, use_cache=False):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            if image_grid_thw is None or input_ids is None:
                raise ValueError("Image inputs require image_grid_thw and input_ids")
            image_embeds = self.visual(pixel_values.to(dtype=self.visual.dtype), image_grid_thw).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if int(image_mask.sum()) != image_embeds.numel():
                raise ValueError("Image features and image placeholder tokens do not match")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            if image_grid_thw is not None and past_length == 0:
                if mm_token_type_ids is None:
                    raise ValueError("Image inputs require processor mm_token_type_ids")
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, mm_token_type_ids, image_grid_thw, attention_mask)
            elif rope_deltas is not None:
                length = inputs_embeds.shape[1]
                if attention_mask is not None:
                    text_positions = attention_mask.long().cumsum(-1) - 1
                    text_positions.masked_fill_(attention_mask == 0, 0)
                    text_positions = text_positions[:, -length:]
                else:
                    text_positions = torch.arange(past_length, past_length + length,
                                                  device=inputs_embeds.device)[None]
                position_ids = text_positions[None].expand(3, -1, -1) + rope_deltas
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                      position_ids=position_ids, past_key_values=past_key_values,
                                      use_cache=use_cache)
        return outputs.last_hidden_state, outputs.past_key_values, rope_deltas


class Flux3PromptUpsampler(TextEncoder[PromptUpsamplerOutput]):
    def __init__(self, config: Flux3PromptUpsamplerConfig):
        super().__init__(config)
        self.model = Flux3PupModel(config.arch_config)
        text = config.arch_config.text_config
        self.lm_head = ReplicatedLinear(text.hidden_size, text.vocab_size, bias=False)

    def forward(self, input_ids=None, logits_to_keep=0, **kwargs):
        hidden, cache, rope_deltas = self.model(input_ids=input_ids, **kwargs)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits, _ = self.lm_head(hidden[:, indices, :])
        return PromptUpsamplerOutput(logits, cache, rope_deltas)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name not in parameters:
                raise ValueError(f"Unexpected PUP weight: {name}")
            if name in loaded:
                raise ValueError(f"Duplicate PUP weight: {name}")
            default_weight_loader(parameters[name], value)
            loaded.add(name)
        missing = parameters.keys() - loaded
        if missing:
            raise ValueError(f"Missing PUP weights: {sorted(missing)}")
        return loaded

    def materialize_non_persistent_buffers(self, device, dtype=None):
        self.model.language_model.materialize_non_persistent_buffers(device, dtype)
        if hasattr(self.model.visual, "materialize_non_persistent_buffers"):
            self.model.visual.materialize_non_persistent_buffers(device, dtype)

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=MAX_NEW_TOKENS, presence_penalty=0.0,
                 stop_token_ids=None, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError("PUP generation accepts one request at a time")
        if max_new_tokens < 1 or input_ids.shape[1] + max_new_tokens > self.config.text_len:
            raise ValueError(f"PUP prompt and generation must fit within {self.config.text_len} tokens")
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        text_positions = attention_mask.long().cumsum(-1) - 1
        text_positions.masked_fill_(attention_mask == 0, 0)
        rope_deltas = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        if kwargs.get("image_grid_thw") is not None:
            vision_positions, rope_deltas = self.model.get_rope_index(
                input_ids, kwargs["mm_token_type_ids"], kwargs["image_grid_thw"], attention_mask)
        else:
            vision_positions = text_positions[None].expand(3, -1, -1)
        position_ids = torch.cat((text_positions[None], vision_positions), dim=0)
        output = self(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                      logits_to_keep=1, use_cache=True, rope_deltas=rope_deltas, **kwargs)
        generated = []
        stopped = set(stop_token_ids or [self.config.eos_token_id, self.config.pad_token_id])
        for step in range(max_new_tokens):
            scores = output.logits[:, -1].float()
            if presence_penalty and generated:
                scores[:, torch.cat(generated, dim=1).unique()] -= presence_penalty
            token = scores.argmax(dim=-1, keepdim=True)
            generated.append(token)
            if token.item() in stopped or step + 1 == max_new_tokens:
                break
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
            position_ids = position_ids[:, :, -1:] + 1
            output = self(input_ids=token, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=output.past_key_values, rope_deltas=rope_deltas,
                          logits_to_keep=1, use_cache=True)
        return torch.cat((input_ids, *generated), dim=1)

    @staticmethod
    def prepare_inputs(prompt, images, boxes, processor, system_prompts):
        from fastvideo.models.encoders.flux3_pup.generation import prepare_inputs
        return prepare_inputs(prompt, images, boxes, processor, system_prompts)

    @torch.inference_mode()
    def upsample(self, prompt, images=None, boxes=None, max_new_tokens=MAX_NEW_TOKENS,
                 processor=None, system_prompts=None):
        from fastvideo.models.encoders.flux3_pup import bbox
        if processor is None or system_prompts is None:
            raise ValueError("PUP requires its processor and system prompts")
        images = images or []
        inputs, locked = self.prepare_inputs(prompt, images, boxes, processor, system_prompts)
        inputs = {key: value.to(self.lm_head.weight.device) for key, value in inputs.items()}
        stop = [processor.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]
        output = self.generate(**inputs, max_new_tokens=max_new_tokens, stop_token_ids=stop,
                               presence_penalty=PRESENCE_PENALTY["i2i" if images else "t2i"])
        caption = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("Empty prompt upsampler generation")
        return bbox.reconcile(caption, locked) if locked else caption


EntryClass = Flux3PromptUpsampler
