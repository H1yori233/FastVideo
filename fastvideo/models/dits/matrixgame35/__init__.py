"""Matrix-Game 3.5 transformer package."""

from .model import MatrixGame35Transformer3DModel, MatrixGame35TransformerBlock

__all__ = [
    "MatrixGame35Transformer3DModel",
    "MatrixGame35TransformerBlock",
]

EntryClass = MatrixGame35Transformer3DModel
