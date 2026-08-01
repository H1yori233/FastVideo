# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.prompts import (
    MatrixGame35TextEncodingStage,
    load_matrixgame35_section_prompts,
    normalize_matrixgame35_section_prompts,
    resolve_matrixgame35_section_prompts,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"


def test_official_six_block_caption_resolves_one_prompt_per_section() -> None:
    caption_path = _OFFICIAL_DIR / "samples" / "distilled" / "suburban_street_6blocks" / "caption.json"
    if not caption_path.is_file():
        pytest.skip(f"Pinned Matrix-Game 3.5 sample is missing: {caption_path}")
    revision = subprocess.run(
        ["git", "-C", str(_OFFICIAL_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_OFFICIAL_REVISION
    detailed = json.loads(caption_path.read_text(encoding="utf-8"))["detailed"]

    prompts = load_matrixgame35_section_prompts(caption_path, num_frames=505)

    assert prompts == [detailed[str(frame)]["dynamic"] for frame in (0, 85, 169, 253, 337, 421)]


def test_caption_resolution_collapses_duplicates_prefers_earlier_ties_and_pads_tail(tmp_path: Path) -> None:
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps({
            "detailed": {
                "0": {
                    "start": "S",
                    "dynamic": "A"
                },
                "30": {
                    "start": "S",
                    "dynamic": "A"
                },
                "43": {
                    "start": "",
                    "dynamic": "B"
                },
                "85": {
                    "start": "",
                    "dynamic": "C"
                },
            }
        }),
        encoding="utf-8",
    )

    assert load_matrixgame35_section_prompts(caption_path, num_frames=253) == ["SA", "C", "C"]


def test_list_and_frame_entry_caption_formats_are_supported(tmp_path: Path) -> None:
    list_path = tmp_path / "segments.json"
    list_path.write_text(
        json.dumps({
            "detailed": [
                {
                    "start": 0,
                    "prompt": {
                        "start": "opening ",
                        "dynamic": "view"
                    },
                },
                {
                    "start": 85,
                    "prompt": {
                        "start": "",
                        "dynamic": "turn"
                    },
                },
            ]
        }),
        encoding="utf-8",
    )
    frame_path = tmp_path / "frames.json"
    frame_path.write_text(
        json.dumps({
            "prompt_cache_format": "frame_entries_v1",
            "detailed": {
                "1": {
                    "dynamic": "left"
                },
                "2": {
                    "dynamic": "right"
                },
            },
        }),
        encoding="utf-8",
    )

    assert load_matrixgame35_section_prompts(list_path, num_frames=169) == ["openingview", "turn"]
    assert load_matrixgame35_section_prompts(frame_path, num_frames=85) == ["right"]


def test_section_prompt_normalization_repeats_scalar_and_rejects_mismatched_lists() -> None:
    assert normalize_matrixgame35_section_prompts("same", None, num_sections=2) == ["same", "same"]
    assert normalize_matrixgame35_section_prompts("fallback", ["first", "second"], num_sections=2) == [
        "first",
        "second",
    ]
    with pytest.raises(ValueError, match="exactly 2 section prompts"):
        normalize_matrixgame35_section_prompts("fallback", ["first"], num_sections=2)
    with pytest.raises(ValueError, match="requires a scalar prompt"):
        normalize_matrixgame35_section_prompts(["batch", "prompt"], None, num_sections=2)


def test_caption_path_resolves_sections_and_rejects_explicit_section_conflict(tmp_path: Path) -> None:
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps({
            "detailed": {
                "0": {
                    "dynamic": "forward"
                },
                "85": {
                    "dynamic": "turn"
                },
            }
        }),
        encoding="utf-8",
    )

    assert resolve_matrixgame35_section_prompts(
        None,
        None,
        str(caption_path),
        num_frames=169,
    ) == ["forward", "turn"]
    with pytest.raises(ValueError, match="either caption_path or section_prompts"):
        resolve_matrixgame35_section_prompts(
            "fallback",
            ["first", "second"],
            str(caption_path),
            num_frames=169,
        )


def test_text_stage_batches_positive_sections_and_encodes_negative_once() -> None:
    stage = MatrixGame35TextEncodingStage(text_encoders=[object()], tokenizers=[object()])
    calls: list[str | list[str]] = []

    def fake_encode_text(self, text, _fastvideo_args, **_kwargs):
        calls.append(text)
        batch_size = len(text) if isinstance(text, list) else 1
        marker = 1.0 if isinstance(text, list) else -1.0
        return [torch.full((batch_size, 2, 3), marker)], [torch.ones(batch_size, 2, dtype=torch.long)]

    stage.encode_text = MethodType(fake_encode_text, stage)
    batch = ForwardBatch(
        data_type="video",
        prompt=None,
        section_prompts=["first", "second"],
        negative_prompt="negative",
        prompt_attention_mask=[],
        negative_attention_mask=[],
        num_frames=169,
        guidance_scale=3.0,
    )
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(text_encoder_configs=[object()]),
        enable_stage_verification=True,
    )

    output = stage(batch, args)

    assert calls == [["first", "second"], "negative"]
    assert output.prompt is None
    assert output.prompt_embeds[0].shape == (2, 2, 3)
    assert output.negative_prompt_embeds[0].shape == (1, 2, 3)
    assert output.prompt_attention_mask[0].shape == (2, 2)
    assert output.negative_attention_mask[0].shape == (1, 2)
