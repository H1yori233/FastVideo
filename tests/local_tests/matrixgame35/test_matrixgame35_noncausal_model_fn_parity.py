# SPDX-License-Identifier: Apache-2.0
"""Tiny direct parity for the released non-causal Matrix-Game 3.5 token path."""

import os
from pathlib import Path

import pytest
import torch

import fastvideo.attention.layer as attention_layer
from fastvideo.attention.backends.sdpa import SDPABackend
from fastvideo.configs.models.dits.matrixgame35 import (
    MatrixGame35WanVideoArchConfig,
    MatrixGame35WanVideoConfig,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.matrixgame35.rope import (
    apply_matrixgame35_rope,
    build_matrixgame35_rope_frequencies,
    matrixgame35_rope_tables,
)
from fastvideo.models.dits.matrixgame35 import MatrixGame35Transformer3DModel
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.pipelines.basic.matrixgame35.layout import build_noncausal_latent_layout
from tests.local_tests.matrixgame35._upstream import (
    load_upstream_pipeline,
    load_upstream_transformer,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = Path(os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", _REPO_ROOT / "Matrix-Game-3.5"))


def _camera_info(frame_count: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    projection = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, frame_count, 4, 1, 1)
    for frame in range(frame_count):
        for subframe in range(4):
            projection[0, frame, subframe, 0, 0] = 1.0 + 0.01 * frame
            projection[0, frame, subframe, 1, 1] = 1.0 + 0.01 * subframe
            projection[0, frame, subframe, 0, 3] = 0.02 * (frame + subframe)
    projection_transpose = projection.transpose(-1, -2).contiguous()
    projection_inverse = torch.linalg.inv(projection)
    return projection, (projection, projection_transpose, projection_inverse)


def _native_model(official: torch.nn.Module) -> MatrixGame35Transformer3DModel:
    arch = MatrixGame35WanVideoArchConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=128,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=256,
        num_layers=1,
        subject_ref_memory_max_refs=2,
    )
    model = MatrixGame35Transformer3DModel(
        MatrixGame35WanVideoConfig(arch_config=arch),
        hf_config={},
    )
    map_name = get_param_names_mapping(model.param_names_mapping)
    mapped_state = {map_name(name)[0]: value for name, value in official.state_dict().items()}
    incompatible = model.load_state_dict(mapped_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    return model.eval()


def test_arbitrary_position_rope_carrier_matches_pinned_complex_math() -> None:
    if not _OFFICIAL_DIR.is_dir():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {_OFFICIAL_DIR}")
    upstream = load_upstream_transformer(_OFFICIAL_DIR).wan_video_dit
    expected_tables = upstream.precompute_freqs_cis_3d(128)[:3]
    actual_tables = matrixgame35_rope_tables(128)
    for actual, expected in zip(actual_tables, expected_tables):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    time_indices = torch.tensor([0, 7, 3, 10, 11])
    actual_frequencies = build_matrixgame35_rope_frequencies(
        actual_tables,
        time_indices,
        height=2,
        width=2,
        device="cpu",
    )
    expected_frequencies = torch.cat(
        (
            expected_tables[0][time_indices].view(5, 1, 1, -1).expand(5, 2, 2, -1),
            expected_tables[1][:2].view(1, 2, 1, -1).expand(5, 2, 2, -1),
            expected_tables[2][:2].view(1, 1, 2, -1).expand(5, 2, 2, -1),
        ),
        dim=-1,
    ).reshape(20, 1, -1)
    torch.testing.assert_close(actual_frequencies, expected_frequencies, rtol=0.0, atol=0.0)

    tensor = torch.randn(1, 20, 1, 128, generator=torch.Generator().manual_seed(3504))
    expected_rotated = upstream.rope_apply(tensor.flatten(2), expected_frequencies, 1).unflatten(2, (1, 128))
    actual_rotated = apply_matrixgame35_rope(tensor, actual_frequencies)
    torch.testing.assert_close(actual_rotated, expected_rotated, rtol=0.0, atol=0.0)


def test_noncausal_clean_mosaic_subject_path_matches_pinned_model_fn_drop_holes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _OFFICIAL_DIR.is_dir():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {_OFFICIAL_DIR}")
    upstream_model = load_upstream_transformer(_OFFICIAL_DIR).wan_video_dit
    upstream_pipeline = load_upstream_pipeline(_OFFICIAL_DIR)

    # This tiny reference model runs on CPU. A CUDA environment can still make
    # the upstream module discover flash-attn at import time, so force its exact
    # built-in SDPA fallback instead of dispatching a CPU tensor to a CUDA-only
    # kernel.
    monkeypatch.setattr(upstream_model, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(upstream_model, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(upstream_model, "SAGE_ATTN_AVAILABLE", False)

    torch.manual_seed(3505)
    official = upstream_model.WanModel(
        dim=128,
        in_dim=4,
        ffn_dim=256,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=1,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        use_prope=True,
        subject_ref_memory_enabled=True,
        subject_ref_memory_max_refs=2,
    ).eval()

    monkeypatch.setattr(
        attention_layer,
        "get_attn_backend",
        lambda *args, **kwargs: SDPABackend,
    )
    native = _native_model(official)

    generator = torch.Generator().manual_seed(3506)
    noisy = torch.randn(1, 4, 2, 4, 4, generator=generator)
    clean = torch.randn(1, 4, 1, 4, 4, generator=generator)
    mosaic = torch.randn(1, 4, 2, 4, 4, generator=generator)
    mosaic[:, :, 0, :2, :2] = 0
    subject = torch.randn(1, 4, 4, 4, generator=generator)
    context = torch.randn(1, 3, 16, generator=generator)
    timestep = torch.tensor([500.0])
    mosaic_indices = torch.tensor([1, 0])
    rope_time_indices = torch.tensor([0, 7, 3, 10, 11])
    camera_info = _camera_info(frame_count=5)

    monkeypatch.setattr(
        upstream_pipeline.WAN_VIDEO_PROPE_CAMERA_UNIT,
        "process",
        lambda **_kwargs: {"camera_info": camera_info},
    )
    with torch.inference_mode():
        expected_noisy = upstream_pipeline.model_fn_wan_video(
            dit=official,
            latents=noisy,
            timestep=timestep,
            context=context,
            first_frame_latents=clean,
            mosaic_latent=mosaic,
            mosaic_frame_indices=mosaic_indices,
            latent_rope_time_indices=rope_time_indices,
            mosaic_mask_holes=True,
            mosaic_drop_holes=True,
            subject_ref_latents=subject,
            subject_ref_slot_ratio=0.5,
            subject_ref_time_gap=2,
            subject_ref_prope_mode="identity",
            height=64,
            width=64,
        )
        expected_no_subject = upstream_pipeline.model_fn_wan_video(
            dit=official,
            latents=noisy,
            timestep=timestep,
            context=context,
            first_frame_latents=clean,
            mosaic_latent=mosaic,
            mosaic_frame_indices=mosaic_indices,
            latent_rope_time_indices=rope_time_indices,
            mosaic_mask_holes=True,
            mosaic_drop_holes=True,
            height=64,
            width=64,
        )

    layout = build_noncausal_latent_layout(
        noisy,
        timestep,
        first_frame_latents=clean,
        mosaic_latents=mosaic,
        mosaic_frame_indices=mosaic_indices,
        latent_rope_time_indices=rope_time_indices,
        subject_ref_prefix_token_count=1,
        drop_mosaic_holes=True,
    )
    assert layout.mosaic_hole_mask is not None
    assert int(layout.mosaic_hole_mask.sum().item()) == 1
    assert layout.latent_rope_time_indices.tolist() == [0, 7, 3, 10, 11]

    block_sequence_lengths: list[int] = []
    hook = native.blocks[0].register_forward_pre_hook(
        lambda _module, args: block_sequence_lengths.append(int(args[0].shape[1])))

    def _run_native(current_layout, current_subject=subject):
        with torch.inference_mode(), set_forward_context(current_timestep=500, attn_metadata=None):
            return native(
                hidden_states=current_layout.latents,
                encoder_hidden_states=context,
                timestep=current_layout.token_timesteps.unsqueeze(0),
                camera_info=camera_info,
                latent_layout=current_layout,
                subject_ref_latents=current_subject,
                subject_ref_slot_ratio=0.5,
                subject_ref_time_gap=2,
                subject_ref_prope_mode="identity",
                height=64,
                width=64,
            )

    try:
        actual_full = _run_native(layout)
        masked_layout = build_noncausal_latent_layout(
            noisy,
            timestep,
            first_frame_latents=clean,
            mosaic_latents=mosaic,
            mosaic_frame_indices=mosaic_indices,
            latent_rope_time_indices=rope_time_indices,
            subject_ref_prefix_token_count=1,
        )
        masked_full = _run_native(masked_layout)

        mosaic_without_holes = mosaic.clone()
        mosaic_without_holes[:, :, 0, :2, :2] = 1
        no_hole_drop_layout = build_noncausal_latent_layout(
            noisy,
            timestep,
            first_frame_latents=clean,
            mosaic_latents=mosaic_without_holes,
            mosaic_frame_indices=mosaic_indices,
            latent_rope_time_indices=rope_time_indices,
            subject_ref_prefix_token_count=1,
            drop_mosaic_holes=True,
        )
        no_hole_mask_layout = build_noncausal_latent_layout(
            noisy,
            timestep,
            first_frame_latents=clean,
            mosaic_latents=mosaic_without_holes,
            mosaic_frame_indices=mosaic_indices,
            latent_rope_time_indices=rope_time_indices,
            subject_ref_prefix_token_count=1,
        )
        no_hole_drop_full = _run_native(no_hole_drop_layout)
        no_hole_mask_full = _run_native(no_hole_mask_layout)
        no_subject_layout = build_noncausal_latent_layout(
            noisy,
            timestep,
            first_frame_latents=clean,
            mosaic_latents=mosaic,
            mosaic_frame_indices=mosaic_indices,
            latent_rope_time_indices=rope_time_indices,
            drop_mosaic_holes=True,
        )
        no_subject_full = _run_native(no_subject_layout, None)
    finally:
        hook.remove()

    assert actual_full.shape == layout.latents.shape
    assert block_sequence_lengths == [20, 21, 21, 21, 19]
    assert torch.count_nonzero(actual_full[:, :, 1, :2, :2]).item() == 0
    actual_noisy = actual_full[:, :, layout.output_frame_slice]
    torch.testing.assert_close(
        actual_noisy,
        masked_full[:, :, masked_layout.output_frame_slice],
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(no_hole_drop_full, no_hole_mask_full, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        no_subject_full[:, :, no_subject_layout.output_frame_slice],
        expected_no_subject,
        rtol=2e-5,
        atol=2e-5,
    )
    max_abs_diff = float((actual_noisy - expected_noisy).abs().max().item())
    print(f"Matrix-Game 3.5 noncausal model_fn max_abs_diff={max_abs_diff:.8g}")
    torch.testing.assert_close(actual_noisy, expected_noisy, rtol=2e-5, atol=2e-5)
