# SPDX-License-Identifier: Apache-2.0
"""State-free causal K/V-cache operations for Matrix-Game 3.5."""

from collections.abc import Sequence
from typing import TypedDict

import torch

from fastvideo.models.dits._matrixgame35_prope import (
    prope_dot_product_attention_by_frame_indices,
)
from fastvideo.models.dits._matrixgame35_rope import apply_matrixgame35_rope


class MatrixGame35CausalKVCache(TypedDict):
    """One block's lazy cache in the released inference layout."""

    k: torch.Tensor | None
    v: torch.Tensor | None
    positions: list[int]
    frames: list[int]
    chunk_ids: list[int]


def init_matrixgame35_causal_kv_caches(
    num_blocks: int,
) -> list[MatrixGame35CausalKVCache]:
    """Create per-block caches whose tensor storage is allocated on first write."""
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
    return [
        {
            "k": None,
            "v": None,
            "positions": [],
            "frames": [],
            "chunk_ids": [],
        }
        for _ in range(num_blocks)
    ]


def _as_int_list(values: Sequence[int] | torch.Tensor, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().reshape(-1).cpu().tolist()
    try:
        return [int(value) for value in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence of integers.") from exc


def _current_chunk_keep_mask(
    chunk_ids: Sequence[int] | torch.Tensor | None,
    *,
    current_token_count: int,
    current_frame_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    if chunk_ids is None:
        return None
    resolved_ids = _as_int_list(chunk_ids, "current_cache_chunk_ids")
    if len(resolved_ids) != current_frame_count:
        raise ValueError(
            "current_cache_chunk_ids length must match current_frames, got "
            f"{len(resolved_ids)} vs {current_frame_count}."
        )
    if current_token_count % current_frame_count:
        raise ValueError("Current token count must be divisible by current frame count.")
    tokens_per_frame = current_token_count // current_frame_count
    token_ids = torch.as_tensor(resolved_ids, device=device, dtype=torch.long).repeat_interleave(tokens_per_frame)
    is_anchor = token_ids < 0
    same_chunk = token_ids[:, None] == token_ids[None, :]
    keep_for_context = same_chunk | is_anchor[None, :]
    keep_for_anchor = is_anchor[None, :].expand_as(keep_for_context)
    return torch.where(is_anchor[:, None], keep_for_anchor, keep_for_context)


def matrixgame35_causal_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    rope_frequencies: torch.Tensor,
    cache_rope_frequencies: torch.Tensor | None,
    viewmats: Sequence[torch.Tensor],
    camera_layout: str,
    mosaic_token_count: int,
    current_positions: Sequence[int] | torch.Tensor,
    current_frames: Sequence[int] | torch.Tensor,
    mosaic_frames: Sequence[int] | torch.Tensor,
    cache: MatrixGame35CausalKVCache,
    cache_frames: Sequence[int] | torch.Tensor,
    cache_read_chunk_id: int | None = None,
    current_cache_chunk_ids: Sequence[int] | torch.Tensor | None = None,
    write_cache: bool = False,
    mosaic_hole_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend the current chunk to cached history, mosaic, and itself.

    ``key`` is cached before native RoPE and ``value`` is cached raw, allowing
    caller-supplied rolling positions to re-anchor cached history on every read.
    """
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must share shape [B, S, H, D].")
    sequence_length = int(query.shape[1])
    mosaic_token_count = int(mosaic_token_count)
    if mosaic_token_count < 0 or mosaic_token_count >= sequence_length:
        raise ValueError(
            "mosaic_token_count must leave a non-empty current chunk, got "
            f"{mosaic_token_count} for sequence {sequence_length}."
        )

    current_positions_list = _as_int_list(current_positions, "current_positions")
    current_frames_list = _as_int_list(current_frames, "current_frames")
    mosaic_frames_list = _as_int_list(mosaic_frames, "mosaic_frames")
    cache_frames_list = _as_int_list(cache_frames, "cache_frames")
    if len(current_positions_list) != len(current_frames_list) or not current_frames_list:
        raise ValueError("current_positions and current_frames must have the same non-zero length.")

    current_token_count = sequence_length - mosaic_token_count
    if current_token_count % len(current_frames_list):
        raise ValueError("Current token count must be divisible by current frame count.")
    tokens_per_frame = current_token_count // len(current_frames_list)
    if mosaic_token_count != len(mosaic_frames_list) * tokens_per_frame:
        raise ValueError("Mosaic token count must match mosaic_frames and the current spatial grid.")

    current_key_pre = key[:, mosaic_token_count:]
    current_value_raw = value[:, mosaic_token_count:]
    rotated_query = apply_matrixgame35_rope(query, rope_frequencies)
    rotated_key = apply_matrixgame35_rope(key, rope_frequencies)

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    key_value_frames: list[int] = []
    keep_parts: list[torch.Tensor] = []

    cached_key_pre = cache.get("k")
    cached_value_raw = cache.get("v")
    if cached_key_pre is not None and int(cached_key_pre.shape[1]) > 0:
        if cached_value_raw is None or cached_value_raw.shape != cached_key_pre.shape:
            raise ValueError("Cached key and value tensors must have matching shapes.")
        if cached_key_pre.ndim != 3 or cached_key_pre.shape[0] != query.shape[0]:
            raise ValueError("Cached key/value must have shape [B, S, hidden_dim].")
        if cached_key_pre.shape[2] != query.shape[2] * query.shape[3]:
            raise ValueError("Cached key/value hidden size does not match the current attention block.")
        if not cache_frames_list or cached_key_pre.shape[1] != len(cache_frames_list) * tokens_per_frame:
            raise ValueError("Cached token count must match cache_frames and the current spatial grid.")
        if cache_rope_frequencies is None or cache_rope_frequencies.shape[0] != cached_key_pre.shape[1]:
            raise ValueError("cache_rope_frequencies must provide one frequency per cached token.")

        selected_key_pre = cached_key_pre
        selected_value_raw = cached_value_raw
        selected_frequencies = cache_rope_frequencies
        cached_chunk_ids = cache.get("chunk_ids")
        if (
            cache_read_chunk_id is not None
            and cached_chunk_ids is not None
            and len(cached_chunk_ids) == len(cache_frames_list)
        ):
            frame_keep = [
                int(chunk_id) < 0 or int(chunk_id) == int(cache_read_chunk_id)
                for chunk_id in cached_chunk_ids
            ]
            if not all(frame_keep):
                token_keep = torch.as_tensor(
                    frame_keep,
                    device=selected_key_pre.device,
                    dtype=torch.bool,
                ).repeat_interleave(tokens_per_frame)
                selected_key_pre = selected_key_pre[:, token_keep]
                selected_value_raw = selected_value_raw[:, token_keep]
                selected_frequencies = selected_frequencies.index_select(
                    0,
                    torch.nonzero(token_keep, as_tuple=False).reshape(-1).to(selected_frequencies.device),
                )
                cache_frames_list = [
                    frame for frame, keep_frame in zip(cache_frames_list, frame_keep) if keep_frame
                ]

        if int(selected_key_pre.shape[1]) > 0:
            selected_key = selected_key_pre.unflatten(2, (query.shape[2], query.shape[3]))
            selected_key = apply_matrixgame35_rope(selected_key, selected_frequencies)
            selected_value = selected_value_raw.unflatten(2, (query.shape[2], query.shape[3]))
            key_parts.append(selected_key)
            value_parts.append(selected_value)
            key_value_frames.extend(cache_frames_list)
            keep_parts.append(
                torch.ones(selected_key.shape[1], device=query.device, dtype=torch.bool)
            )

    if mosaic_token_count:
        key_parts.append(rotated_key[:, :mosaic_token_count])
        value_parts.append(value[:, :mosaic_token_count])
        key_value_frames.extend(mosaic_frames_list)
        if mosaic_hole_keep is None:
            mosaic_keep = torch.ones(mosaic_token_count, device=query.device, dtype=torch.bool)
        else:
            mosaic_keep = mosaic_hole_keep.to(device=query.device, dtype=torch.bool).reshape(-1)
            if mosaic_keep.shape != (mosaic_token_count,):
                raise ValueError("mosaic_hole_keep must have one value per mosaic token.")
        keep_parts.append(mosaic_keep)

    current_key_start = sum(int(part.shape[1]) for part in key_parts)
    key_parts.append(rotated_key[:, mosaic_token_count:])
    value_parts.append(value[:, mosaic_token_count:])
    key_value_frames.extend(current_frames_list)
    keep_parts.append(torch.ones(current_token_count, device=query.device, dtype=torch.bool))

    keep = torch.cat(keep_parts)
    attention_mask = None
    current_keep = _current_chunk_keep_mask(
        current_cache_chunk_ids,
        current_token_count=current_token_count,
        current_frame_count=len(current_frames_list),
        device=query.device,
    )
    if current_keep is not None:
        query_keep = keep.view(1, -1).expand(current_token_count, -1).clone()
        query_keep[:, current_key_start:current_key_start + current_token_count] &= current_keep
        if not bool(query_keep.all()):
            attention_mask = query_keep.view(1, 1, current_token_count, -1)
    elif not bool(keep.all()):
        attention_mask = keep.view(1, 1, 1, -1)

    output = torch.zeros_like(query)
    output[:, mosaic_token_count:] = prope_dot_product_attention_by_frame_indices(
        rotated_query[:, mosaic_token_count:].transpose(1, 2),
        torch.cat(key_parts, dim=1).transpose(1, 2),
        torch.cat(value_parts, dim=1).transpose(1, 2),
        viewmats=viewmats,
        query_frame_indices=current_frames_list,
        key_value_frame_indices=key_value_frames,
        camera_layout=camera_layout,
        attn_mask=attention_mask,
    ).transpose(1, 2)

    if write_cache:
        new_key = current_key_pre.flatten(2).detach()
        new_value = current_value_raw.flatten(2).detach()
        if current_cache_chunk_ids is None:
            new_chunk_ids = [-1] * len(current_frames_list)
        else:
            new_chunk_ids = _as_int_list(current_cache_chunk_ids, "current_cache_chunk_ids")
            if len(new_chunk_ids) != len(current_frames_list):
                raise ValueError("current_cache_chunk_ids length must match current_frames.")
        if cached_key_pre is None or int(cached_key_pre.shape[1]) == 0:
            cache["k"] = new_key
            cache["v"] = new_value
            cache["positions"] = list(current_positions_list)
            cache["frames"] = list(current_frames_list)
            cache["chunk_ids"] = list(new_chunk_ids)
        else:
            cache["k"] = torch.cat((cached_key_pre, new_key), dim=1)
            cache["v"] = torch.cat((cached_value_raw, new_value), dim=1)
            cache["positions"] = list(cache.get("positions", cache.get("frames", []))) + current_positions_list
            cache["frames"] = list(cache.get("frames", [])) + current_frames_list
            cache["chunk_ids"] = list(cache.get("chunk_ids", [])) + new_chunk_ids

    return output
