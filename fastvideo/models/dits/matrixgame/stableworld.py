from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from fastvideo.logger import init_logger

try:
    import cv2
except ImportError:  # pragma: no cover - exercised through fallback tests
    cv2 = None

logger = init_logger(__name__)

STABLEWORLD_SUBFRAME_INDEX = 2
STABLEWORLD_EVICT_OLDEST = 0
STABLEWORLD_EVICT_MIDDLE = 1
_STABLEWORLD_DEFAULT_FRAMES_PER_LATENT = 4
_STABLEWORLD_FIRST_CHUNK_FRAMES = 1


@dataclass
class StableWorldDecodedChunk:
    latent_start: int
    num_latent_frames: int
    frames: torch.Tensor
    is_first_chunk: bool = False


@dataclass
class StableWorldRuntimeState:
    threshold: float
    window_ids: list[int]
    decoded_chunks: list[StableWorldDecodedChunk]
    preview_cache: list[torch.Tensor | None]
    evict_policy: int = STABLEWORLD_EVICT_OLDEST
    last_decoded_chunk: torch.Tensor | None = None


def stableworld_is_available() -> bool:
    return cv2 is not None


def chw_to_gray_u8(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().cpu().float()

    if x.min() < 0:
        x = ((x + 1.0) * 127.5).clamp(0, 255.0)
    else:
        x = (x * 255.0).clamp(0, 255.0)

    if x.shape[0] == 3:
        r, g, b = x[0], x[1], x[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = x[0]

    return gray.byte().numpy()


def orb_ransac_score_chw(
    chw_a: torch.Tensor,
    chw_b: torch.Tensor,
    ratio_thresh: float = 0.8,
    min_good: int = 30,
) -> float:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for StableWorld eviction")

    img1 = chw_to_gray_u8(chw_a)
    img2 = chw_to_gray_u8(chw_b)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    keypoints1, desc1 = orb.detectAndCompute(img1, None)
    keypoints2, desc2 = orb.detectAndCompute(img2, None)
    if desc1 is None or desc2 is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for matches in knn_matches:
        if len(matches) < 2:
            continue
        match_a, match_b = matches
        if match_a.distance < ratio_thresh * match_b.distance:
            good_matches.append(match_a)

    if len(good_matches) < 7:
        return 0.0

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    _, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    _, mask_f = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        3.0,
        0.99,
    )

    inliers_h = int(mask_h.sum()) if mask_h is not None else 0
    inliers_f = (
        int(mask_f.sum())
        if isinstance(mask_f, np.ndarray) and mask_f.size > 0
        else 0
    )

    denom = max(len(good_matches), 1)
    ratio_h = inliers_h / denom
    ratio_f = inliers_f / denom
    score = max(ratio_h, ratio_f)

    if len(good_matches) < min_good:
        score *= len(good_matches) / float(min_good)

    return float(score)


def _decoded_frame_counts(
    chunk: StableWorldDecodedChunk,
) -> list[int]:
    total_frames = int(chunk.frames.shape[2])
    if chunk.num_latent_frames <= 0:
        return []

    counts = [
        _STABLEWORLD_DEFAULT_FRAMES_PER_LATENT
        for _ in range(chunk.num_latent_frames)
    ]
    if chunk.is_first_chunk:
        counts[0] = _STABLEWORLD_FIRST_CHUNK_FRAMES

    counts[-1] += total_frames - sum(counts)
    if any(count <= 0 for count in counts):
        raise ValueError(
            "StableWorld decoded frame spans must stay positive, got "
            f"{counts} for total_frames={total_frames}"
        )
    return counts


def get_decoded_frame_by_latent(
    chunks: list[StableWorldDecodedChunk],
    latent_id: int,
    sub: int = STABLEWORLD_SUBFRAME_INDEX,
    batch_index: int = 0,
) -> torch.Tensor:
    for chunk in chunks:
        chunk_end = chunk.latent_start + chunk.num_latent_frames
        if not chunk.latent_start <= latent_id < chunk_end:
            continue

        local_latent_idx = latent_id - chunk.latent_start
        counts = _decoded_frame_counts(chunk)
        frame_start = sum(counts[:local_latent_idx])
        frame_count = counts[local_latent_idx]
        frame_offset = frame_start + min(sub, frame_count - 1)
        return chunk.frames[batch_index, :, frame_offset]

    raise ValueError(f"StableWorld could not resolve latent_id={latent_id}")


def _fallback_window(
    window_ids: list[int],
    num_frame_per_block: int,
) -> tuple[int, list[int], float]:
    if len(window_ids) < num_frame_per_block:
        raise ValueError(
            "StableWorld window must contain at least num_frame_per_block latent ids"
        )
    last_id = window_ids[-1]
    next_ids = [
        last_id + offset
        for offset in range(1, num_frame_per_block + 1)
    ]
    return (
        STABLEWORLD_EVICT_OLDEST,
        window_ids[num_frame_per_block:] + next_ids,
        0.0,
    )


def decide_and_update_window_ids(
    window_ids: list[int],
    decoded_chunks: list[StableWorldDecodedChunk],
    sim_threshold: float,
    num_frame_per_block: int = 3,
    sub: int = STABLEWORLD_SUBFRAME_INDEX,
    batch_index: int = 0,
    similarity_fn: Callable[[torch.Tensor, torch.Tensor], float]
    | None = None,
) -> tuple[int, list[int], float]:
    if len(window_ids) < num_frame_per_block:
        raise ValueError(
            "StableWorld window must contain at least num_frame_per_block latent ids"
        )

    compare_index_1 = num_frame_per_block - 1
    compare_index_2 = compare_index_1 + num_frame_per_block

    if compare_index_2 >= len(window_ids) or not decoded_chunks:
        return _fallback_window(window_ids, num_frame_per_block)

    if similarity_fn is None:
        if not stableworld_is_available():
            logger.warning(
                "StableWorld requested but OpenCV is unavailable; "
                "falling back to FIFO eviction"
            )
            return _fallback_window(window_ids, num_frame_per_block)
        similarity_fn = orb_ransac_score_chw

    latent_id_1 = window_ids[compare_index_1]
    latent_id_2 = window_ids[compare_index_2]

    try:
        frame_1 = get_decoded_frame_by_latent(
            decoded_chunks,
            latent_id_1,
            sub=sub,
            batch_index=batch_index,
        )
        frame_2 = get_decoded_frame_by_latent(
            decoded_chunks,
            latent_id_2,
            sub=sub,
            batch_index=batch_index,
        )
    except ValueError:
        return _fallback_window(window_ids, num_frame_per_block)

    similarity = float(similarity_fn(frame_1, frame_2))
    last_id = window_ids[-1]
    next_ids = [
        last_id + offset
        for offset in range(1, num_frame_per_block + 1)
    ]

    if similarity >= sim_threshold:
        kept = (
            window_ids[:num_frame_per_block]
            + window_ids[2 * num_frame_per_block:]
        )
        evict_policy = STABLEWORLD_EVICT_MIDDLE
    else:
        kept = window_ids[num_frame_per_block:]
        evict_policy = STABLEWORLD_EVICT_OLDEST

    new_window_ids = kept + next_ids
    if len(new_window_ids) != len(window_ids):
        raise AssertionError(
            "StableWorld window length changed unexpectedly: "
            f"{len(window_ids)} -> {len(new_window_ids)}"
        )

    return evict_policy, new_window_ids, similarity
