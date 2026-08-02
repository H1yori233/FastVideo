# SPDX-License-Identifier: Apache-2.0
"""Model-owned K/V-cache operations for Matrix-Game 3.5 causal attention."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

Cache = dict[str, Any]


def copy_causal_kv_caches(caches: Sequence[Cache] | None) -> list[Cache]:
    """Copy cache containers without copying the immutable tensor payloads."""
    return [{
        "k": cache.get("k"),
        "v": cache.get("v"),
        "positions": list(cache.get("positions", cache.get("frames", []))),
        "frames": list(cache.get("frames", [])),
        "chunk_ids": list(cache.get("chunk_ids", [-1] * len(cache.get("frames", [])))),
    } for cache in (caches or [])]


def concat_causal_kv_caches(left: Sequence[Cache] | None, right: Sequence[Cache] | None) -> list[Cache]:
    """Concatenate two per-block cache lists at frame granularity."""
    if not left:
        return copy_causal_kv_caches(right)
    if not right:
        return copy_causal_kv_caches(left)
    if len(left) != len(right):
        raise ValueError(f"K/V cache block count mismatch: {len(left)} vs {len(right)}.")

    output: list[Cache] = []
    for left_cache, right_cache in zip(left, right, strict=False):
        left_k, left_v = left_cache.get("k"), left_cache.get("v")
        right_k, right_v = right_cache.get("k"), right_cache.get("v")
        if left_k is None or int(left_k.shape[1]) == 0:
            key, value = right_k, right_v
        elif right_k is None or int(right_k.shape[1]) == 0:
            key, value = left_k, left_v
        else:
            key = torch.cat((left_k, right_k), dim=1)
            value = torch.cat((left_v, right_v), dim=1)
        left_frames = list(left_cache.get("frames", []))
        right_frames = list(right_cache.get("frames", []))
        output.append({
            "k":
            key,
            "v":
            value,
            "positions":
            list(left_cache.get("positions", left_frames)) + list(right_cache.get("positions", right_frames)),
            "frames":
            left_frames + right_frames,
            "chunk_ids":
            list(left_cache.get("chunk_ids", [-1] * len(left_frames))) +
            list(right_cache.get("chunk_ids", [-1] * len(right_frames))),
        })
    return output


def causal_kv_frame_count(caches: Sequence[Cache] | None) -> int:
    """Return the common frame count carried by a cache list."""
    return 0 if not caches else len(caches[0].get("frames", []))


def slice_causal_kv_cache_frames(
    caches: Sequence[Cache] | None,
    frame_indices: Sequence[int],
    *,
    context: str,
) -> list[Cache]:
    """Slice cache tensors and metadata by logical frame indices."""
    indices = [int(index) for index in frame_indices]
    if not caches:
        return []

    output: list[Cache] = []
    for block_index, cache in enumerate(caches):
        key, value = cache.get("k"), cache.get("v")
        frames = list(cache.get("frames", []))
        positions = list(cache.get("positions", frames))
        chunk_ids = list(cache.get("chunk_ids", [-1] * len(frames)))
        frame_count = len(frames)
        if frame_count != len(positions) or frame_count != len(chunk_ids):
            raise ValueError(f"{context}: cache metadata mismatch at block {block_index}: "
                             f"frames={frame_count}, positions={len(positions)}, chunk_ids={len(chunk_ids)}.")
        if key is None or value is None or int(key.shape[1]) == 0 or frame_count == 0:
            output.append({"k": key, "v": value, "positions": [], "frames": [], "chunk_ids": []})
            continue
        if int(key.shape[1]) % frame_count:
            raise ValueError(f"{context}: K/V token count {int(key.shape[1])} is not divisible by "
                             f"frame count {frame_count} at block {block_index}.")
        tokens_per_frame = int(key.shape[1]) // frame_count
        token_indices: list[int] = []
        for frame_index in indices:
            if frame_index < 0 or frame_index >= frame_count:
                raise IndexError(f"{context}: frame index {frame_index} outside [0, {frame_count}).")
            start = frame_index * tokens_per_frame
            token_indices.extend(range(start, start + tokens_per_frame))
        if token_indices:
            tensor_indices = torch.as_tensor(token_indices, device=key.device, dtype=torch.long)
            selected_key = key.index_select(1, tensor_indices).contiguous()
            selected_value = value.index_select(1, tensor_indices).contiguous()
        else:
            selected_key = key[:, :0].contiguous()
            selected_value = value[:, :0].contiguous()
        output.append({
            "k": selected_key,
            "v": selected_value,
            "positions": [positions[index] for index in indices],
            "frames": [frames[index] for index in indices],
            "chunk_ids": [chunk_ids[index] for index in indices],
        })
    return output


def tail_causal_kv_cache_frames(
    caches: Sequence[Cache] | None,
    frame_count: int,
    *,
    context: str,
) -> list[Cache]:
    """Return the final ``frame_count`` logical frames."""
    frame_count = int(frame_count)
    total = causal_kv_frame_count(caches)
    if frame_count <= 0:
        return slice_causal_kv_cache_frames(caches, [], context=context)
    if frame_count > total:
        raise ValueError(f"{context}: requested {frame_count} tail frames from a {total}-frame cache.")
    return slice_causal_kv_cache_frames(caches, range(total - frame_count, total), context=context)


def trim_causal_kv_rolling_window(
    caches: Sequence[Cache] | None,
    *,
    frames_per_chunk: int = 3,
    window_chunks: int = 7,
) -> list[Cache]:
    """Keep the advancing boundary anchor plus six three-frame chunks.

    Positions and camera-frame ids stay global. Eviction changes visibility but
    never renumbers retained frames.
    """
    frames_per_chunk = int(frames_per_chunk)
    window_chunks = int(window_chunks)
    if frames_per_chunk <= 0 or window_chunks <= 0:
        raise ValueError("frames_per_chunk and window_chunks must be positive.")
    max_frames = 1 + (window_chunks - 1) * frames_per_chunk
    output = copy_causal_kv_caches(caches)
    while causal_kv_frame_count(output) > max_frames:
        total = causal_kv_frame_count(output)
        if total <= frames_per_chunk:
            return tail_causal_kv_cache_frames(output, max_frames, context="distilled rolling-cache fallback")
        # Before eviction: [anchor, oldest three-frame chunk, newer chunks...].
        # The oldest chunk's final frame becomes the next boundary anchor.
        keep = [frames_per_chunk, *range(1 + frames_per_chunk, total)]
        output = slice_causal_kv_cache_frames(output, keep, context="distilled rolling-cache trim")
    return output


__all__ = [
    "causal_kv_frame_count",
    "concat_causal_kv_caches",
    "copy_causal_kv_caches",
    "slice_causal_kv_cache_frames",
    "tail_causal_kv_cache_frames",
    "trim_causal_kv_rolling_window",
]
