# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from types import ModuleType

import pytest
import torch

from fastvideo.pipelines.basic.matrixgame35.layout import build_noncausal_latent_layout
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


PARITY_SCOPE = "implementation_subcomponent"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_SOURCE = _REPO_ROOT / "Matrix-Game-3.5" / "diffsynth" / "pipelines" / "wan_video.py"


class _PipelineUnit:
    def __init__(self, **kwargs) -> None:
        self.input_params = kwargs.get("input_params")
        self.output_params = kwargs.get("output_params")


def _load_upstream_layout_helpers() -> ModuleType:
    """Execute the selected real helpers from the pinned upstream source."""
    if not _UPSTREAM_SOURCE.is_file():
        pytest.skip(f"Pinned upstream source is missing: {_UPSTREAM_SOURCE}")
    revision = subprocess.run(
        ["git", "-C", str(_UPSTREAM_SOURCE.parents[2]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_OFFICIAL_REVISION:
        raise RuntimeError(
            "Matrix-Game 3.5 reference revision mismatch: "
            f"expected {PINNED_OFFICIAL_REVISION}, got {revision}"
        )

    source_tree = ast.parse(_UPSTREAM_SOURCE.read_text(), filename=str(_UPSTREAM_SOURCE))
    selected_names = {
        "_resolve_mosaic_frame_indices",
        "_resolve_latent_rope_time_indices",
        "_build_mosaic_cross_attn_keep_mask",
        "WanVideoUnit_LatentSequence",
    }
    selected_nodes = [
        node
        for node in source_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected_names
    ]
    if {node.name for node in selected_nodes} != selected_names:
        raise RuntimeError("Pinned upstream latent-layout helpers are incomplete.")

    module = ModuleType("matrixgame35_upstream_layout")
    module.__dict__.update(torch=torch, PipelineUnit=_PipelineUnit)
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(_UPSTREAM_SOURCE), "exec"), module.__dict__)
    return module


def test_sequence_indices_and_rope_times_match_pinned_upstream() -> None:
    upstream = _load_upstream_layout_helpers()
    noisy = torch.arange(1 * 2 * 4 * 4 * 6, dtype=torch.float32).reshape(1, 2, 4, 4, 6)
    first = torch.full((1, 2, 2, 4, 6), 10.0, dtype=torch.float64)
    mosaic = torch.full((1, 2, 2, 4, 6), 20.0, dtype=torch.float64)
    mosaic_indices = [1, 3]

    expected = upstream.WanVideoUnit_LatentSequence().process(
        pipe=None,
        latents=noisy,
        first_frame_latents=first,
        mosaic_latent=mosaic,
        mosaic_frame_indices=mosaic_indices,
    )
    actual = build_noncausal_latent_layout(
        noisy,
        500.0,
        first_frame_latents=first,
        mosaic_latents=mosaic,
        mosaic_frame_indices=mosaic_indices,
    )
    expected_rope_times = upstream._resolve_latent_rope_time_indices(
        None,
        first_frame_count=expected["first_frame_count"],
        mosaic_frame_count=expected["mosaic_frame_count"],
        noisy_frame_count=noisy.shape[2],
        mosaic_frame_indices=expected["mosaic_frame_indices"],
        device=noisy.device,
    )

    torch.testing.assert_close(actual.latents, expected["latents"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.mosaic_frame_indices, expected["mosaic_frame_indices"])
    torch.testing.assert_close(actual.latent_rope_time_indices, expected_rope_times)
    assert actual.latent_rope_time_indices.tolist() == [0, 1, 3, 5, 2, 3, 4, 5]
    assert actual.condition_frame_count == expected["condition_frame_count"] == 4


def test_timesteps_and_cross_attention_mask_match_upstream_layout() -> None:
    upstream = _load_upstream_layout_helpers()
    noisy = torch.ones(1, 2, 3, 4, 4)
    first = torch.ones(1, 2, 1, 4, 4)
    mosaic = torch.ones(1, 2, 2, 4, 4)
    actual = build_noncausal_latent_layout(
        noisy,
        torch.tensor([417.0]),
        first_frame_latents=first,
        mosaic_latents=mosaic,
        mosaic_frame_indices=(0, 2),
        subject_ref_prefix_token_count=3,
    )
    expected_mask = upstream._build_mosaic_cross_attn_keep_mask(
        prefix_memory_token_count=3,
        reference_token_count=0,
        first_frame_count=1,
        mosaic_frame_count=2,
        noisy_frame_count=3,
        tokens_per_frame=4,
        device=noisy.device,
    )

    expected_timesteps = torch.tensor([0.0] * 12 + [417.0] * 12)
    torch.testing.assert_close(actual.token_timesteps, expected_timesteps, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.cross_attention_keep_mask, expected_mask)
    assert not actual.cross_attention_keep_mask[:3].any()
    assert not actual.cross_attention_keep_mask[7:15].any()


def test_hole_detection_uses_full_1x2x2_patch_and_overrides_timestep_to_1000() -> None:
    noisy = torch.ones(1, 2, 2, 4, 4)
    first = torch.ones(1, 2, 1, 4, 4)
    mosaic = torch.zeros(1, 2, 1, 4, 4)
    mosaic[0, 0, 0, 0, 1] = 1.0  # Non-top-left content keeps the whole first patch.
    actual = build_noncausal_latent_layout(
        noisy,
        333.0,
        first_frame_latents=first,
        mosaic_latents=mosaic,
        mosaic_frame_indices=(0,),
    )

    expected_holes = torch.tensor([False] * 4 + [False, True, True, True] + [False] * 8)
    expected_timesteps = torch.tensor([0.0] * 4 + [0.0, 1000.0, 1000.0, 1000.0] + [333.0] * 8)
    torch.testing.assert_close(actual.mosaic_hole_mask, expected_holes)
    torch.testing.assert_close(actual.token_timesteps, expected_timesteps, rtol=0.0, atol=0.0)


def test_output_slices_and_explicit_rope_override() -> None:
    noisy = torch.ones(1, 2, 2, 4, 4)
    actual = build_noncausal_latent_layout(noisy, 250.0, latent_rope_time_indices=(7, 9))

    assert actual.output_frame_slice == slice(0, 2)
    assert actual.output_token_slice == slice(0, 8)
    assert actual.cross_attention_keep_mask is None
    assert actual.mosaic_hole_mask is None
    assert actual.latent_rope_time_indices.tolist() == [7, 9]


def test_rejects_noncanonical_layouts() -> None:
    noisy = torch.ones(1, 2, 3, 4, 4)
    with pytest.raises(ValueError, match="mosaic_frame_indices is required"):
        build_noncausal_latent_layout(noisy, 1.0, mosaic_latents=torch.ones(1, 2, 1, 4, 4))
    with pytest.raises(ValueError, match="even latent height and width"):
        build_noncausal_latent_layout(torch.ones(1, 2, 3, 3, 4), 1.0)
    with pytest.raises(ValueError, match="sequence_parallel_size=1"):
        build_noncausal_latent_layout(noisy, 1.0, sequence_parallel_size=2)
