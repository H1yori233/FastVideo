# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 UMT5 configuration."""

from dataclasses import dataclass

from fastvideo.configs.models.encoders.t5 import T5ArchConfig


@dataclass
class MatrixGame35T5ArchConfig(T5ArchConfig):
    """Keep the released fixed-length tokenizer contract after HF updates."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs["padding"] = "max_length"


__all__ = ["MatrixGame35T5ArchConfig"]
