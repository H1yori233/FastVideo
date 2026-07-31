# SPDX-License-Identifier: Apache-2.0
"""Section prompt handling for Matrix-Game 3.5 rollouts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.matrixgame35.camera import RGB_FRAMES_PER_BLOCK
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

MATRIXGAME35_NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                                "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
                                "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
                                "杂乱的背景，三条腿，背景人很多，倒着走")


def normalize_matrixgame35_section_prompts(
    prompt: str | list[str] | None,
    section_prompts: list[str] | None,
    *,
    num_sections: int,
) -> list[str]:
    """Return exactly one prompt per 84-frame Matrix-Game section."""
    if not isinstance(num_sections, int) or isinstance(num_sections, bool) or num_sections <= 0:
        raise ValueError(f"num_sections must be a positive integer, got {num_sections!r}.")
    if section_prompts is None:
        if not isinstance(prompt, str):
            raise ValueError("Matrix-Game 3.5 requires a scalar prompt when section_prompts is not provided.")
        return [prompt] * num_sections
    if not isinstance(section_prompts, list) or not all(isinstance(value, str) for value in section_prompts):
        raise ValueError("Matrix-Game 3.5 section_prompts must be a list of strings.")
    if len(section_prompts) != num_sections:
        raise ValueError(
            f"Matrix-Game 3.5 requires exactly {num_sections} section prompts, got {len(section_prompts)}.")
    return list(section_prompts)


def _compose_prompt(start: Any, dynamic: Any) -> str:
    start_text = str(start or "").strip()
    dynamic_text = str(dynamic or "").strip()
    return "".join(part for part in (start_text, dynamic_text) if part).strip()


def _load_prompt_payload(path: Path) -> tuple[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read Matrix-Game 3.5 caption JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Matrix-Game 3.5 caption JSON must contain an object: {path}")
    detailed = payload.get("detailed")
    if not isinstance(detailed, dict | list) or not detailed:
        raise ValueError(f"Matrix-Game 3.5 caption JSON has no detailed prompts: {path}")
    if payload.get("prompt_cache_format") == "frame_entries_v1":
        if not isinstance(detailed, dict):
            raise ValueError("frame_entries_v1 captions require a numeric-keyed detailed object.")
        frames: dict[int, str] = {}
        try:
            keys = sorted(detailed, key=int)
        except (TypeError, ValueError) as error:
            raise ValueError("Matrix-Game 3.5 detailed prompt keys must be integer frame indices.") from error
        for key in keys:
            entry = detailed[key]
            if isinstance(entry, dict):
                text = _compose_prompt(entry.get("start"), entry.get("dynamic"))
                if text:
                    frames[int(key)] = text
        return "frames", frames

    segments: list[tuple[int, str, str]] = []
    if isinstance(detailed, list):
        for item in detailed:
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), dict):
                continue
            try:
                start_frame = int(item["start"])
            except (KeyError, TypeError, ValueError):
                continue
            entry = item["prompt"]
            start = str(entry.get("start") or "").strip()
            dynamic = str(entry.get("dynamic") or "").strip()
            if start or dynamic:
                segments.append((start_frame, start, dynamic))
        segments.sort(key=lambda item: item[0])
    else:
        try:
            keys = sorted(detailed, key=int)
        except (TypeError, ValueError) as error:
            raise ValueError("Matrix-Game 3.5 detailed prompt keys must be integer frame indices.") from error
        previous: tuple[str, str] | None = None
        for key in keys:
            entry = detailed[key]
            if not isinstance(entry, dict):
                continue
            value = (
                str(entry.get("start") or "").strip(),
                str(entry.get("dynamic") or "").strip(),
            )
            if value != previous:
                segments.append((int(key), *value))
                previous = value
    if not segments:
        raise ValueError(f"Matrix-Game 3.5 caption JSON has no usable prompt segments: {path}")
    return "segments", segments


def _resolve_segment_prompt(segments: list[tuple[int, str, str]], frame_min: int, frame_max: int) -> str:

    def segment_index(frame: int) -> int:
        index = 0
        for candidate, (start_frame, _start, _dynamic) in enumerate(segments):
            if start_frame <= frame:
                index = candidate
            else:
                break
        return index

    first_index = segment_index(frame_min)
    last_index = segment_index(frame_max)
    best_index = first_index
    best_overlap = -1
    for index in range(first_index, last_index + 1):
        start_frame = segments[index][0]
        next_start = segments[index + 1][0] if index + 1 < len(segments) else frame_max + 1
        overlap = max(0, min(frame_max + 1, next_start) - max(frame_min, start_frame))
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    _start_frame, start, dynamic = segments[best_index]
    return _compose_prompt(start, dynamic)


def load_matrixgame35_section_prompts(
    caption_path: str | Path,
    *,
    num_frames: int,
    fallback_prompt: str | None = None,
) -> list[str]:
    """Resolve official caption JSON into one prompt per generation section."""
    if (not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 1
            or (num_frames - 1) % RGB_FRAMES_PER_BLOCK):
        raise ValueError("Matrix-Game 3.5 requires num_frames = 1 + 84 * num_sections.")
    format_name, entries = _load_prompt_payload(Path(caption_path))
    prompts: list[str] = []
    for section_index in range((num_frames - 1) // RGB_FRAMES_PER_BLOCK):
        frame_min = 1 + section_index * RGB_FRAMES_PER_BLOCK
        frame_max = (section_index + 1) * RGB_FRAMES_PER_BLOCK
        if format_name == "frames":
            values = [entries.get(frame) for frame in range(frame_min, frame_max + 1) if entries.get(frame)]
            prompt = max(Counter(values).items(), key=lambda item: (item[1], item[0]))[0] if values else ""
        else:
            prompt = _resolve_segment_prompt(entries, frame_min, frame_max)
        if not prompt:
            if fallback_prompt is None:
                raise ValueError(f"Matrix-Game 3.5 caption has no prompt for RGB frames {frame_min}..{frame_max}.")
            prompt = fallback_prompt
        prompts.append(prompt)
    return prompts


def resolve_matrixgame35_section_prompts(
    prompt: str | list[str] | None,
    section_prompts: list[str] | None,
    caption_path: str | None,
    *,
    num_frames: int,
) -> list[str]:
    """Resolve either explicit sections or the released caption JSON path."""
    num_sections = (num_frames - 1) // RGB_FRAMES_PER_BLOCK
    if caption_path is not None:
        if section_prompts is not None:
            raise ValueError("Provide either caption_path or section_prompts, not both.")
        fallback = prompt if isinstance(prompt, str) else None
        return load_matrixgame35_section_prompts(
            caption_path,
            num_frames=num_frames,
            fallback_prompt=fallback,
        )
    return normalize_matrixgame35_section_prompts(
        prompt,
        section_prompts,
        num_sections=num_sections,
    )


class MatrixGame35TextEncodingStage(TextEncodingStage):
    """Encode all positive section prompts once and one shared CFG negative."""

    @staticmethod
    def _validate_embeddings(batch: ForwardBatch, num_sections: int) -> None:
        if len(batch.prompt_embeds) != 1:
            raise ValueError("Matrix-Game 3.5 requires exactly one text encoder embedding tensor.")
        positive = batch.prompt_embeds[0]
        if positive.ndim != 3 or positive.shape[0] != num_sections:
            raise ValueError("Matrix-Game 3.5 positive text embeddings must have shape "
                             f"[{num_sections}, sequence, hidden], got {tuple(positive.shape)}.")
        if batch.do_classifier_free_guidance:
            if batch.negative_prompt_embeds is None or len(batch.negative_prompt_embeds) != 1:
                raise ValueError("Matrix-Game 3.5 CFG requires exactly one negative text embedding tensor.")
            negative = batch.negative_prompt_embeds[0]
            if negative.ndim != 3 or negative.shape[0] != 1 or negative.shape[1:] != positive.shape[1:]:
                raise ValueError("Matrix-Game 3.5 negative text embeddings must have shape "
                                 f"[1, {positive.shape[1]}, {positive.shape[2]}], got {tuple(negative.shape)}.")

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        num_sections = (batch.num_frames - 1) // RGB_FRAMES_PER_BLOCK
        batch.section_prompts = normalize_matrixgame35_section_prompts(
            batch.prompt,
            batch.section_prompts,
            num_sections=num_sections,
        )
        if batch.prompt_embeds:
            self._validate_embeddings(batch, num_sections)
            return batch

        all_indices = list(range(len(self.text_encoders)))
        prompt_embeds, prompt_masks = self.encode_text(
            batch.section_prompts,
            fastvideo_args,
            encoder_index=all_indices,
            return_attention_mask=True,
            max_length=batch.max_sequence_length,
        )
        batch.prompt_embeds.extend(prompt_embeds)
        if batch.prompt_attention_mask is not None:
            batch.prompt_attention_mask.extend(prompt_masks)

        if batch.do_classifier_free_guidance:
            if not isinstance(batch.negative_prompt, str):
                raise ValueError("Matrix-Game 3.5 CFG requires a scalar negative prompt.")
            negative_embeds, negative_masks = self.encode_text(
                batch.negative_prompt,
                fastvideo_args,
                encoder_index=all_indices,
                return_attention_mask=True,
                max_length=batch.max_sequence_length,
            )
            assert batch.negative_prompt_embeds is not None
            batch.negative_prompt_embeds.extend(negative_embeds)
            if batch.negative_attention_mask is not None:
                batch.negative_attention_mask.extend(negative_masks)

        self._validate_embeddings(batch, num_sections)
        return batch


__all__ = [
    "MATRIXGAME35_NEGATIVE_PROMPT",
    "MatrixGame35TextEncodingStage",
    "load_matrixgame35_section_prompts",
    "normalize_matrixgame35_section_prompts",
    "resolve_matrixgame35_section_prompts",
]
