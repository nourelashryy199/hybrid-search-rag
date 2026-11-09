
from typing import Protocol

from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.domain.retrieval import SearchResult


class DocumentRepository(Protocol):
    def get(self, document_id: str) -> IndexedDocument | None:
        ...


class VectorRetriever(Protocol):
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        ...


class VersionFilteredVectorRetriever:
    def __init__(
        self,
        retriever: VectorRetriever,
        repository: DocumentRepository,
        candidate_k: int = 100,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        self.retriever = retriever
        self.repository = repository
        self.candidate_k = candidate_k

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidates = self.retriever.search(
            query_vector,
            top_k=max(top_k, self.candidate_k),
        )

        active_results = []

        for result in candidates:
            chunk = result.chunk

            indexed_document = self.repository.get(
                chunk.document_id
            )

            if indexed_document is None:
                continue

            if chunk.chunk_id not in indexed_document.chunk_ids:
                continue

            active_results.append(result)

            if len(active_results) == top_k:
                break

        return active_results
