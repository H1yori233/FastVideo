"""Matrix-Game 3.5 pipeline entrypoints."""

from .matrixgame35_base_first_person_pipeline import MatrixGame35BaseFirstPersonPipeline
from .matrixgame35_base_third_person_pipeline import MatrixGame35BaseThirdPersonPipeline
from .matrixgame35_distilled_first_person_pipeline import MatrixGame35DistilledFirstPersonPipeline

__all__ = [
    "MatrixGame35BaseFirstPersonPipeline",
    "MatrixGame35BaseThirdPersonPipeline",
    "MatrixGame35DistilledFirstPersonPipeline",
]
