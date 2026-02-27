# SPDX-License-Identifier: Apache-2.0
"""
Immutable KV cache for MatrixGame causal inference.

Replaces the mutable dict + rolling eviction pattern with a simpler
concat + slice design. Since official MatrixGame configs all use
sink_size=0, sink token support is not needed.
"""

import torch


class KVCache:
    """Sliding-window KV cache with immutable update semantics.
    """

    def __init__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int = 0,
        window_size: int | None = None,
    ):
        self.k = k
        self.v = v
        self.length = length
        self.window_size = window_size if window_size is not None else k.shape[1]

    @staticmethod
    def create(
        batch_size: int,
        window_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "KVCache":
        k = torch.zeros(
            [batch_size, window_size, num_heads, head_dim],
            dtype=dtype,
            device=device,
        )
        v = torch.zeros(
            [batch_size, window_size, num_heads, head_dim],
            dtype=dtype,
            device=device,
        )
        return KVCache(k, v, length=0, window_size=window_size)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> "KVCache":
        self.k = torch.cat([self.k[:, :self.length], new_k], dim=1)[:, -self.window_size:]
        self.v = torch.cat([self.v[:, :self.length], new_v], dim=1)[:, -self.window_size:]
        self.length = min(self.length + new_k.shape[1], self.window_size)
        return self

    def get_attended_kv(
        self,
        max_attn_size: int | None = None,
        repeat_factor: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_attn_size is not None:
            start = max(0, self.length - max_attn_size)
        else:
            start = max(0, self.window_size - self.length)

        cached_k = self.k[:, start:start + min(self.length, max_attn_size or self.length)]
        cached_v = self.v[:, start:start + min(self.length, max_attn_size or self.length)]

        if repeat_factor is not None:
            cached_k = cached_k.repeat(repeat_factor, 1, 1, 1)
            cached_v = cached_v.repeat(repeat_factor, 1, 1, 1)

        return cached_k, cached_v

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.length = 0
