
import math

from hybrid_rag.domain.reranking import (
    RerankedResult,
    Reranker,
)
from hybrid_rag.domain.retrieval import SearchResult


class RerankingService:
    """Rerank retrieved chunks using a configurable model."""

    def __init__(self, reranker: Reranker) -> None:
        self.reranker = reranker

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not results:
            return []

        chunk_ids = [
            result.chunk.chunk_id
            for result in results
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Duplicate chunk IDs in retrieval results"
            )

        chunks = [
            result.chunk
            for result in results
        ]

        scores = self.reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k,
        )

        if len(scores) != len(chunks):
            raise ValueError(
                "Reranker must return one score per input chunk"
            )

        if any(not math.isfinite(score) for score in scores):
            raise ValueError(
                "Reranker returned a non-finite score"
            )

        reranked = [
            RerankedResult(
                chunk=chunk,
                score=score,
                original_rank=index + 1,
            )
            for index, (chunk, score) in enumerate(
                zip(chunks, scores)
            )
        ]

        # Deterministic ordering for equal scores.
        reranked.sort(
            key=lambda result: (
                -result.score,
                result.original_rank,
            )
        )

        return reranked[:top_k]
