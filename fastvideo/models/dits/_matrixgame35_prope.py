# SPDX-License-Identifier: Apache-2.0
"""Parameter-free full-layout PRoPE attention for Matrix-Game 3.5.

The helper operates on per-rank ``[batch, heads, sequence, head_dim]``
tensors and calls Torch SDPA directly. Sequence-parallel integration belongs
to the owning attention module.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange


_FULL_CAMERA_LAYOUT = "full"
_PROJECTION_BLOCK_SIZE = 4 * 4 * 4


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    viewmats: Sequence[torch.Tensor],
    camera_layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if camera_layout != _FULL_CAMERA_LAYOUT:
        raise ValueError(
            "Matrix-Game 3.5 supports only camera_layout='full', "
            f"got {camera_layout!r}."
        )
    if len(viewmats) != 3:
        raise ValueError(
            "viewmats must contain exactly (projection, transpose, inverse)."
        )
    if not all(isinstance(matrix, torch.Tensor) for matrix in viewmats):
        raise ValueError("viewmats must contain tensors.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, sequence, head_dim].")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {key.shape} and {value.shape}.")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have matching batch, head, and head_dim dimensions.")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device.")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype.")

    projection, transpose, inverse = viewmats
    expected_matrix_shape = projection.shape
    if projection.ndim != 5 or projection.shape[-3:] != (4, 4, 4):
        raise ValueError(
            "projection matrices must have shape [batch, camera_frames, 4, 4, 4], "
            f"got {projection.shape}."
        )
    if transpose.shape != expected_matrix_shape or inverse.shape != expected_matrix_shape:
        raise ValueError("projection, transpose, and inverse matrix shapes must match.")
    if projection.shape[0] != query.shape[0]:
        raise ValueError(
            f"viewmat batch {projection.shape[0]} does not match query batch {query.shape[0]}."
        )
    if projection.shape[1] == 0:
        raise ValueError("viewmats must contain at least one camera frame.")
    for name, matrix in zip(("projection", "transpose", "inverse"), viewmats):
        if matrix.device != query.device:
            raise ValueError(f"{name} matrix must be on the same device as query.")
        if matrix.dtype != query.dtype:
            raise ValueError(f"{name} matrix must have the same dtype as query.")

    head_dim = query.shape[-1]
    if head_dim == 0 or head_dim % _PROJECTION_BLOCK_SIZE != 0:
        raise ValueError(
            f"head_dim must be a positive multiple of {_PROJECTION_BLOCK_SIZE}, got {head_dim}."
        )
    camera_frames = projection.shape[1]
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.shape[2] % camera_frames != 0:
            raise ValueError(
                f"{name} sequence length {tensor.shape[2]} must be divisible by "
                f"camera frame count {camera_frames}."
            )
    return projection, transpose, inverse


def _apply_tiled_projection(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    projected = rearrange(
        features,
        "b h (s t) (p q j r) -> b h s t r p q j",
        s=matrix.shape[1],
        p=4,
        q=4,
        j=4,
    )
    projected = torch.einsum("bscij,bhstrpcj->bhstrpci", matrix, projected)
    return rearrange(projected, "b h s t r p q j -> b h (s t) (p q j r)")


def apply_matrixgame35_prope_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    viewmats: Sequence[torch.Tensor],
    camera_layout: str = _FULL_CAMERA_LAYOUT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the full-layout camera transforms to query, key, and value."""
    _, transpose, inverse = _validate_inputs(query, key, value, viewmats, camera_layout)
    return (
        _apply_tiled_projection(query, transpose),
        _apply_tiled_projection(key, inverse),
        _apply_tiled_projection(value, inverse),
    )


def prope_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    viewmats: Sequence[torch.Tensor],
    camera_layout: str = _FULL_CAMERA_LAYOUT,
    scale: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """Apply full-layout PRoPE before and after scaled dot-product attention."""
    projected_query, projected_key, projected_value = apply_matrixgame35_prope_qkv(
        query,
        key,
        value,
        viewmats=viewmats,
        camera_layout=camera_layout,
    )
    projection = viewmats[0]
    output = F.scaled_dot_product_attention(
        projected_query,
        projected_key,
        projected_value,
        scale=scale,
        **kwargs,
    )
    output = _apply_tiled_projection(output, projection)
    if output.shape != query.shape:
        raise RuntimeError(f"PRoPE attention returned shape {output.shape}, expected {query.shape}.")
    return output
