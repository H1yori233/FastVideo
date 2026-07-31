# SPDX-License-Identifier: Apache-2.0
"""Model-local conditioning primitives for Matrix-Game 3.5."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

_SUBJECT_REF_ATTRIBUTES = (
    "subject_ref_index_embedding",
    "subject_ref_type_embedding",
    "subject_ref_local_h_embedding",
    "subject_ref_local_w_embedding",
)


def build_mosaic_cross_attention_keep_mask(
    *,
    prefix_memory_token_count: int = 0,
    reference_token_count: int,
    first_frame_count: int,
    mosaic_frame_count: int,
    noisy_frame_count: int,
    tokens_per_frame: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Keep text cross-attention only for ordinary video tokens."""
    prefix_memory_token_count = int(prefix_memory_token_count)
    total = prefix_memory_token_count + int(reference_token_count) + (int(first_frame_count) + int(mosaic_frame_count) +
                                                                      int(noisy_frame_count)) * int(tokens_per_frame)
    mask = torch.ones(total, dtype=torch.bool, device=device)
    if prefix_memory_token_count > 0:
        mask[:prefix_memory_token_count] = False
    if mosaic_frame_count > 0:
        start = (prefix_memory_token_count + int(reference_token_count) +
                 int(first_frame_count) * int(tokens_per_frame))
        end = start + int(mosaic_frame_count) * int(tokens_per_frame)
        mask[start:end] = False
    return mask


def _patch_subject_references(dit: Any, refs: torch.Tensor) -> torch.Tensor:
    patchify = getattr(dit, "patchify", None)
    if callable(patchify):
        return patchify(refs)
    patch_embedding = getattr(dit, "patch_embedding", None)
    if callable(patch_embedding):
        return patch_embedding(refs)
    raise ValueError("subject-reference conditioning requires a patchify or patch_embedding callable.")


def _subject_ref_local_pos(
    table: torch.Tensor,
    length: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    table = table.to(device=device, dtype=dtype)
    length = int(length)
    if length <= int(table.shape[0]):
        return table[:length]
    positions = F.interpolate(
        table.float().transpose(0, 1).unsqueeze(0),
        size=length,
        mode="linear",
        align_corners=True,
    )
    return positions.squeeze(0).transpose(0, 1).to(dtype=dtype)


def _build_subject_ref_time_freqs(
    frequencies: Sequence[torch.Tensor] | None,
    *,
    ref_count: int,
    slot_h: int,
    slot_w: int,
    subject_ref_time_gap: int,
    device: torch.device | str,
) -> torch.Tensor:
    if frequencies is None or len(frequencies) != 3:
        raise ValueError("subject-reference conditioning requires three native RoPE frequency tables.")
    freq_device = frequencies[0].device
    time_gap = max(1, int(subject_ref_time_gap))
    ref_time_indices = (torch.arange(1, ref_count + 1, device=freq_device, dtype=torch.long) *
                        time_gap).clamp(max=int(frequencies[0].shape[0]) - 1)
    # References live at negative temporal positions while remaining at the
    # spatial origin. Complex conjugation flips only the temporal RoPE phase.
    time_freqs = frequencies[0][ref_time_indices].conj()
    h_freqs = frequencies[1][:1].expand(slot_h, -1)
    w_freqs = frequencies[2][:1].expand(slot_w, -1)
    ref_freqs = torch.cat(
        (
            time_freqs.view(ref_count, 1, 1, -1).expand(ref_count, slot_h, slot_w, -1),
            h_freqs.view(1, slot_h, 1, -1).expand(ref_count, slot_h, slot_w, -1),
            w_freqs.view(1, 1, slot_w, -1).expand(ref_count, slot_h, slot_w, -1),
        ),
        dim=-1,
    )
    return ref_freqs.reshape(ref_count * slot_h * slot_w, 1, -1).to(device)


def build_subject_ref_memory_tokens(
    dit: Any,
    subject_ref_latents: Any | None,
    *,
    batch_size: int,
    video_h: int,
    video_w: int,
    subject_ref_slot_ratio: float,
    subject_ref_time_gap: int,
    device: torch.device | str,
    dtype: torch.dtype,
    rope_frequencies: Sequence[torch.Tensor] | None = None,
) -> dict[str, Any] | None:
    """Patch and position subject references as self-attention prefix tokens.

    ``rope_frequencies`` permits a shared arbitrary-position RoPE carrier; when
    omitted, models exposing the three native complex ``freqs`` tables are used.
    """
    if subject_ref_latents is None:
        return None
    if not getattr(dit, "subject_ref_memory_enabled", False):
        return None

    missing_attributes = [name for name in _SUBJECT_REF_ATTRIBUTES if not hasattr(dit, name)]
    if missing_attributes:
        raise ValueError("subject_ref_latents were provided, but the DiT has no "
                         f"{missing_attributes}. Enable subject ref memory before loading.")

    refs = subject_ref_latents
    if not torch.is_tensor(refs):
        refs = torch.as_tensor(refs)
    if refs.ndim == 5:
        # Materialized references are R,C,1,H,W; batched inference is B,C,R,H,W.
        if int(refs.shape[2]) == 1 and int(refs.shape[0]) != 1:
            refs = refs.permute(2, 1, 0, 3, 4).contiguous()
        elif int(refs.shape[0]) in (1, int(batch_size)):
            refs = refs.contiguous()
        else:
            raise ValueError("subject_ref_latents expects (R,C,1,H,W) or (1,C,R,H,W), got "
                             f"{tuple(refs.shape)}.")
    elif refs.ndim == 4:
        refs = refs.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    else:
        raise ValueError(f"subject_ref_latents expects 4 or 5 dims, got {tuple(refs.shape)}.")

    if int(refs.shape[0]) not in (1, int(batch_size)):
        raise ValueError("subject_ref_latents batch size must be 1 or match model batch, got "
                         f"{int(refs.shape[0])} vs {int(batch_size)}.")
    ref_count = int(refs.shape[2])
    if ref_count <= 0:
        return None
    max_refs = int(dit.subject_ref_index_embedding.shape[0])
    if ref_count > max_refs:
        refs = refs[:, :, :max_refs]
        ref_count = max_refs

    refs = refs.to(device=device, dtype=dtype)
    ref_x = _patch_subject_references(dit, refs)
    if ref_x.ndim != 5:
        raise ValueError(f"subject reference patch embedding must return [B,C,R,H,W], got {tuple(ref_x.shape)}.")
    if int(ref_x.shape[0]) == 1 and int(batch_size) > 1:
        ref_x = ref_x.expand(int(batch_size), -1, -1, -1, -1)
    _, _, _, ref_h, ref_w = ref_x.shape

    ratio = min(1.0, max(0.01, float(subject_ref_slot_ratio)))
    slot_size = int(round(min(int(video_h), int(video_w)) * ratio))
    if slot_size <= 0:
        return None
    slot_h = max(1, int(round(ref_h * slot_size / float(video_h))))
    slot_w = max(1, int(round(ref_w * slot_size / float(video_w))))
    slot_h = min(slot_h, ref_h)
    slot_w = min(slot_w, ref_w)
    h_start = int(ref_h - slot_h)
    w_start = int(ref_w - slot_w)

    # The bottom-right crop is the subject slot used by third-person checkpoints.
    ref_x = ref_x[:, :, :, h_start:ref_h, w_start:ref_w]
    ref_x = ref_x.permute(0, 2, 3, 4, 1).contiguous()

    ref_index_pos = dit.subject_ref_index_embedding[:ref_count].to(device=device, dtype=ref_x.dtype)
    ref_type_pos = dit.subject_ref_type_embedding.to(device=device, dtype=ref_x.dtype)
    local_h_pos = _subject_ref_local_pos(
        dit.subject_ref_local_h_embedding,
        slot_h,
        device=device,
        dtype=ref_x.dtype,
    )
    local_w_pos = _subject_ref_local_pos(
        dit.subject_ref_local_w_embedding,
        slot_w,
        device=device,
        dtype=ref_x.dtype,
    )
    local_pos = local_h_pos.view(1, 1, slot_h, 1, -1) + local_w_pos.view(1, 1, 1, slot_w, -1)
    ref_x = (ref_x + ref_type_pos.view(1, 1, 1, 1, -1) + ref_index_pos.view(1, ref_count, 1, 1, -1) + local_pos)
    ref_x = ref_x.reshape(ref_x.shape[0], ref_count * slot_h * slot_w, ref_x.shape[-1]).contiguous()

    ref_freqs = _build_subject_ref_time_freqs(
        rope_frequencies if rope_frequencies is not None else getattr(dit, "freqs", None),
        ref_count=ref_count,
        slot_h=slot_h,
        slot_w=slot_w,
        subject_ref_time_gap=subject_ref_time_gap,
        device=device,
    )
    return {
        "x": ref_x,
        "freqs": ref_freqs,
        "token_count": int(ref_x.shape[1]),
        "ref_count": ref_count,
        "slot_grid": (int(slot_h), int(slot_w)),
        "slot_start": (int(h_start), int(w_start)),
    }


def prepend_subject_ref_prope_camera_info(
    camera_info: tuple[Any, ...] | None,
    *,
    prefix_token_count: int,
    tokens_per_frame: int,
    frame_count: int | None = None,
    mode: str = "identity",
    clean_anchor_token_index: int | None = None,
) -> tuple[Any, ...] | None:
    """Prepend reference-token PRoPE matrices using identity or an anchor view."""
    if camera_info is None or int(prefix_token_count) <= 0:
        return camera_info
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info

    w2c_info = camera_info[0]
    viewmats = camera_info[1]
    view_change_positions = camera_info[2] if len(camera_info) > 2 else None
    projection, projection_transpose, projection_inverse = viewmats
    camera_batch = int(projection.shape[0])
    token_count = int(prefix_token_count)
    mode = str(mode or "identity").strip().lower()
    if mode not in {"identity", "clean_anchor"}:
        raise ValueError(f"subject_ref_prope_mode must be 'identity' or 'clean_anchor', got {mode!r}.")

    def _as_token_viewmats(matrix: torch.Tensor) -> torch.Tensor:
        rest = matrix.shape[2:]
        if frame_count is not None and int(matrix.shape[1]) == int(frame_count):
            return (matrix.unsqueeze(2).expand(camera_batch, int(matrix.shape[1]), int(tokens_per_frame),
                                               *rest).reshape(camera_batch,
                                                              int(matrix.shape[1]) * int(tokens_per_frame),
                                                              *rest).contiguous())
        if frame_count is not None and int(matrix.shape[1]) == int(frame_count) * int(tokens_per_frame):
            return matrix
        return matrix

    def _prepend_refs(matrix: torch.Tensor) -> torch.Tensor:
        token_viewmats = _as_token_viewmats(matrix)
        if mode == "clean_anchor":
            if int(token_viewmats.shape[1]) <= 0:
                raise ValueError("Cannot use subject_ref_prope_mode='clean_anchor' with empty PROPE viewmats.")
            anchor_index = int(clean_anchor_token_index or 0)
            if anchor_index < 0 or anchor_index >= int(token_viewmats.shape[1]):
                raise ValueError("subject_ref_prope_mode='clean_anchor' anchor token index "
                                 f"{anchor_index} is outside PROPE token length {int(token_viewmats.shape[1])}.")
            prefix = token_viewmats[:, anchor_index:anchor_index + 1].expand(
                camera_batch,
                token_count,
                *token_viewmats.shape[2:],
            )
        else:
            identity = torch.eye(
                token_viewmats.shape[-1],
                device=token_viewmats.device,
                dtype=token_viewmats.dtype,
            )
            view_shape = ((1, 1) + tuple(1 for _ in token_viewmats.shape[2:-2]) +
                          (token_viewmats.shape[-2], token_viewmats.shape[-1]))
            prefix = identity.view(view_shape).expand(
                camera_batch,
                token_count,
                *token_viewmats.shape[2:],
            )
        return torch.cat((prefix, token_viewmats), dim=1).contiguous()

    token_viewmats = (
        _prepend_refs(projection),
        _prepend_refs(projection_transpose),
        _prepend_refs(projection_inverse),
    )
    if view_change_positions is None:
        return (w2c_info, token_viewmats)

    if mode == "clean_anchor":
        if int(view_change_positions.shape[1]) <= 0:
            raise ValueError("Cannot use subject_ref_prope_mode='clean_anchor' with empty PROPE view_change_positions.")
        anchor_index = int(clean_anchor_token_index or 0)
        if anchor_index < 0 or anchor_index >= int(view_change_positions.shape[1]):
            raise ValueError("subject_ref_prope_mode='clean_anchor' anchor token index "
                             f"{anchor_index} is outside PROPE view-change token length "
                             f"{int(view_change_positions.shape[1])}.")
        prefix_view_change = view_change_positions[:, anchor_index:anchor_index + 1].expand(
            camera_batch,
            token_count,
            3,
        )
    else:
        prefix_view_change = torch.zeros(
            camera_batch,
            token_count,
            3,
            device=view_change_positions.device,
            dtype=view_change_positions.dtype,
        )
        prefix_view_change[..., 0] = 1.0
    view_change_positions = torch.cat((prefix_view_change, view_change_positions), dim=1).contiguous()
    return (w2c_info, token_viewmats, view_change_positions)
