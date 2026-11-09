
from typing import Protocol

from hybrid_rag.domain.reranking import RerankedResult
from hybrid_rag.domain.retrieval import SearchResult


class HybridRetriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[SearchResult]:
        ...


class ResultReranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        ...


class RetrievalPipeline:
    """Orchestrate hybrid retrieval and optional reranking."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: ResultReranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RerankedResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        # Retrieve enough candidates for the reranker.
        retrieval_limit = (
            candidate_k
            if self.reranker is not None
            else top_k
        )

        results = self.retriever.search(
            query=query,
            top_k=retrieval_limit,
            candidate_k=candidate_k,
        )

        if not results:
            return []

        if self.reranker is not None:
            return self.reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
            )

        # Without a reranker, retain the original RRF ranking.
        return [
            RerankedResult(
                chunk=result.chunk,
                score=result.score,
                original_rank=index + 1,
            )
            for index, result in enumerate(results[:top_k])
        ]
