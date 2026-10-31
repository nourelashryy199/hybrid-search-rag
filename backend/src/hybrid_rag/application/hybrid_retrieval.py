
from typing import Protocol

from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.domain.embedding import EmbeddingModel
from hybrid_rag.domain.retrieval import SearchResult


class VectorRetriever(Protocol):
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        ...


class KeywordRetriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        ...


class HybridRetrievalService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        fusion: ReciprocalRankFusion,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.fusion = fusion

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        query_vector = self.embedding_model.embed_query(query)

        dense_results = self.vector_retriever.search(
            query_vector,
            top_k=candidate_k,
        )

        keyword_results = self.keyword_retriever.search(
            query,
            top_k=candidate_k,
        )

        return self.fusion.fuse(
            rankings=[dense_results, keyword_results],
            top_k=top_k,
        )
