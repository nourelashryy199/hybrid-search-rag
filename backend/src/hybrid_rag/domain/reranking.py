
import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from hybrid_rag.domain.document import DocumentChunk


class RerankedResult(BaseModel):
    """A document chunk with a reranker relevance score."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float
    original_rank: int

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Reranking score must be finite")
        return value

    @field_validator("original_rank")
    @classmethod
    def validate_original_rank(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Original rank must be positive")
        return value


class Reranker(Protocol):
    """Interface implemented by external or local rerankers."""

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int,
    ) -> list[float]:
        """Return one relevance score per input chunk, in input order."""
        ...
