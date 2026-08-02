# SPDX-License-Identifier: Apache-2.0
"""User-facing subject-reference preparation for Matrix-Game 3.5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN_RGB = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


@dataclass(frozen=True)
class SubjectReference:
    """One saved subject cutout and its optional foreground mask."""

    image_path: Path
    mask_path: Path | None = None


def _read_candidates(path: Path) -> list[SubjectReference]:
    references: list[SubjectReference] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}.") from error
            if row.get("frame_idx") is None:
                continue
            image = row.get("image_path")
            if not image:
                raise ValueError(f"Subject reference row {line_number} is missing image_path.")
            mask = row.get("mask_path")
            if not mask:
                raise ValueError(f"Subject reference row {line_number} is missing mask_path.")
            references.append(SubjectReference(
                image_path=path.parent / str(image),
                mask_path=path.parent / str(mask),
            ))
    return references


def _derive_selection_seed(dataset_seed: int, seed_role: str, seed_key: str) -> int:
    blob = b"|".join(str(part).encode() for part in (dataset_seed, seed_role, seed_key))
    return int.from_bytes(hashlib.blake2b(blob, digest_size=4).digest(), "little")


def _sample_reference_indices(
    candidate_count: int,
    pairwise_similarity: np.ndarray,
    *,
    num_refs: int,
    rng: np.random.Generator,
    dissimilar_top_k: int,
    max_similarity: float,
) -> list[int]:
    """Match the released pairwise-dissimilar reference sampler exactly."""
    if candidate_count <= 0 or num_refs <= 0:
        return []
    count = min(int(num_refs), int(candidate_count))
    if count <= 1:
        return [int(rng.integers(0, candidate_count))]

    similarity = np.asarray(pairwise_similarity, dtype=np.float32)
    if similarity.shape != (candidate_count, candidate_count):
        return [int(value) for value in rng.choice(candidate_count, size=count, replace=False)]
    similarity = np.nan_to_num(similarity, nan=1.0, posinf=1.0, neginf=-1.0)
    similarity = np.clip(similarity, -1.0, 1.0)

    selected = [int(rng.integers(0, candidate_count))]
    remaining = set(range(candidate_count))
    remaining.discard(selected[0])
    top_k = max(1, int(dissimilar_top_k))
    threshold = float(max_similarity)
    while remaining and len(selected) < count:
        scored = [(max(float(similarity[int(index), int(chosen)]) for chosen in selected), int(index))
                  for index in sorted(remaining)]
        scored.sort(key=lambda item: item[0])
        valid = [index for similarity_score, index in scored if similarity_score <= threshold]
        if valid:
            choice_pool = valid[:min(top_k, len(valid))]
        else:
            choice_pool = [index for _, index in scored[:min(top_k, len(scored))]]
        choice = int(choice_pool[int(rng.integers(0, len(choice_pool)))])
        selected.append(choice)
        remaining.discard(choice)
    return selected


def _select_reference_indices(
    candidate_count: int,
    pairwise_similarity: np.ndarray | None,
    *,
    max_refs: int,
    dataset_seed: int = 42,
    seed_role: str = "validation_subject_ref",
    seed_key: str = "case",
    dissimilar_top_k: int = 8,
    max_similarity: float = 0.94,
    shuffle_refs: bool = True,
) -> list[int]:
    """Select exported candidate indices using the released validation RNG."""
    if candidate_count <= 0 or max_refs <= 0:
        return []
    rng = np.random.default_rng(_derive_selection_seed(dataset_seed, seed_role, seed_key))
    count = min(int(max_refs), int(candidate_count))
    if pairwise_similarity is None:
        selected = [int(value) for value in rng.choice(candidate_count, size=count, replace=False)]
    else:
        selected = _sample_reference_indices(
            candidate_count,
            pairwise_similarity,
            num_refs=count,
            rng=rng,
            dissimilar_top_k=dissimilar_top_k,
            max_similarity=max_similarity,
        )
    if shuffle_refs and len(selected) > 1:
        selected = [int(value) for value in rng.permutation(selected).tolist()]
    return selected


def _load_pairwise_similarity(source: Path) -> np.ndarray | None:
    path = source / "pairwise_similarity.npy"
    if not path.is_file():
        return None
    try:
        return np.load(path).astype(np.float32, copy=False)
    except Exception:  # noqa: BLE001 - the released loader treats unreadable similarity data as absent.
        return None


def discover_subject_references(source: str | Path) -> list[SubjectReference]:
    """Discover refs from an official export or a directory of plain images."""
    source = Path(source)
    if not source.is_dir():
        raise ValueError(f"subject_ref_source must be a directory, got {source}.")
    candidates = source / "candidates.jsonl"
    if candidates.is_file():
        references = _read_candidates(candidates)
    else:
        images = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES and "_mask" not in path.stem.lower())
        references = []
        for image in images:
            mask = image.with_name(f"{image.stem}_mask.png")
            references.append(SubjectReference(image, mask if mask.is_file() else None))
    if not references:
        raise ValueError(f"No subject reference images found in {source}.")
    for reference in references:
        if not reference.image_path.is_file():
            raise FileNotFoundError(f"Subject reference image does not exist: {reference.image_path}.")
        if reference.mask_path is not None and not reference.mask_path.is_file():
            raise FileNotFoundError(f"Subject reference mask does not exist: {reference.mask_path}.")
    return references


def select_subject_references(
    source: str | Path,
    *,
    max_refs: int,
    dataset_seed: int = 42,
    seed_role: str = "validation_subject_ref",
    seed_key: str = "case",
    dissimilar_top_k: int = 8,
    max_similarity: float = 0.94,
    shuffle_refs: bool = True,
) -> list[SubjectReference]:
    """Match the released public wrapper's deterministic validation selection."""
    if max_refs <= 0:
        return []
    source = Path(source)
    references = discover_subject_references(source)
    selected_indices = _select_reference_indices(
        len(references),
        _load_pairwise_similarity(source) if (source / "candidates.jsonl").is_file() else None,
        max_refs=max_refs,
        dataset_seed=dataset_seed,
        seed_role=seed_role,
        seed_key=seed_key,
        dissimilar_top_k=dissimilar_top_k,
        max_similarity=max_similarity,
        shuffle_refs=shuffle_refs,
    )
    return [references[index] for index in selected_indices]


def _canvas_background(name: str) -> np.ndarray:
    name = str(name).strip().lower()
    if name == "zero":
        return np.asarray((127.5, 127.5, 127.5), dtype=np.float32)
    if name == "black":
        return np.zeros(3, dtype=np.float32)
    if name != "imagenet_mean":
        raise ValueError("subject reference background must be 'imagenet_mean', 'zero', or 'black'.")
    return IMAGENET_MEAN_RGB


def build_subject_reference_canvas(
    reference: SubjectReference,
    *,
    height: int,
    width: int,
    slot_ratio: float = 0.5,
    background: str = "imagenet_mean",
) -> torch.Tensor:
    """Place one official cutout in the bottom-right square canvas slot."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}.")
    if not 0.05 <= float(slot_ratio) <= 1.0:
        raise ValueError(f"slot_ratio must be in [0.05, 1.0], got {slot_ratio}.")

    with Image.open(reference.image_path) as image:
        image_rgb = np.asarray(image.convert("RGB")).copy()
    if reference.mask_path is not None:
        # The released inference consumes foreground-enhanced cutouts. The mask
        # remains part of the export contract and is validated here, but is not
        # applied a second time when building the canvas.
        with Image.open(reference.mask_path) as mask_image:
            mask_image.convert("L").load()

    slot_size = min(max(1, round(min(height, width) * float(slot_ratio))), height, width)
    try:
        # Match upstream exactly when OpenCV is available without making it a
        # package-wide dependency for callers that never use subject refs.
        import cv2

        resized = cv2.resize(image_rgb, (slot_size, slot_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    except ImportError:
        resized = np.asarray(
            Image.fromarray(image_rgb).resize((slot_size, slot_size), Image.Resampling.BOX),
            dtype=np.float32,
        )
    canvas = np.empty((height, width, 3), dtype=np.float32)
    canvas[:] = _canvas_background(background)
    canvas[height - slot_size:, width - slot_size:] = resized
    return torch.from_numpy(canvas).permute(2, 0, 1).mul(2.0 / 255.0).sub(1.0).clamp(-1.0, 1.0)


def load_subject_reference_canvases(
    source: str | Path,
    *,
    height: int,
    width: int,
    max_refs: int,
    slot_ratio: float = 0.5,
    background: str = "imagenet_mean",
    dataset_seed: int = 42,
    seed_role: str = "validation_subject_ref",
    seed_key: str = "case",
    dissimilar_top_k: int = 8,
    max_similarity: float = 0.94,
    shuffle_refs: bool = True,
) -> torch.Tensor:
    """Load at most ``max_refs`` refs as normalized ``[R,3,H,W]`` canvases."""
    if max_refs <= 0:
        return torch.empty((0, 3, height, width), dtype=torch.float32)
    references = select_subject_references(
        source,
        max_refs=max_refs,
        dataset_seed=dataset_seed,
        seed_role=seed_role,
        seed_key=seed_key,
        dissimilar_top_k=dissimilar_top_k,
        max_similarity=max_similarity,
        shuffle_refs=shuffle_refs,
    )
    return torch.stack([
        build_subject_reference_canvas(
            reference,
            height=height,
            width=width,
            slot_ratio=slot_ratio,
            background=background,
        ) for reference in references
    ])
