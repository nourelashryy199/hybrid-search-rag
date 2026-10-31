
import pytest

from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import (
    HybridRetrievalService,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult


def make_result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="test-document",
            text=f"Passage {chunk_id}",
            chunk_index=0,
            page_number=1,
        ),
        score=score,
    )


class FakeEmbeddingModel:
    dimensions = 3

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorRetriever:
    def __init__(self) -> None:
        self.last_top_k = None

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.last_top_k = top_k

        return [
            make_result("a", 0.9),
            make_result("b", 0.8),
        ][:top_k]


class FakeKeywordRetriever:
    def __init__(self) -> None:
        self.last_top_k = None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.last_top_k = top_k

        return [
            make_result("b", 12.0),
            make_result("c", 8.0),
        ][:top_k]


def make_service():
    vector = FakeVectorRetriever()
    keyword = FakeKeywordRetriever()

    service = HybridRetrievalService(
        embedding_model=FakeEmbeddingModel(),
        vector_retriever=vector,
        keyword_retriever=keyword,
        fusion=ReciprocalRankFusion(),
    )

    return service, vector, keyword


def test_hybrid_search_prioritizes_shared_results():
    service, _, _ = make_service()

    results = service.search("example question")

    assert results[0].chunk.chunk_id == "b"


def test_hybrid_search_preserves_source_metadata():
    service, _, _ = make_service()

    results = service.search("example question")

    assert results[0].chunk.document_id == "test-document"
    assert results[0].chunk.page_number == 1


def test_hybrid_search_respects_top_k():
    service, _, _ = make_service()

    results = service.search("example question", top_k=2)

    assert len(results) == 2


def test_hybrid_search_passes_candidate_k():
    service, vector, keyword = make_service()

    service.search(
        "example question",
        top_k=2,
        candidate_k=10,
    )

    assert vector.last_top_k == 10
    assert keyword.last_top_k == 10


def test_hybrid_search_rejects_blank_query():
    service, _, _ = make_service()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.search("   ")


def test_hybrid_search_rejects_invalid_top_k():
    service, _, _ = make_service()

    with pytest.raises(ValueError, match="top_k"):
        service.search("example question", top_k=0)


def test_hybrid_search_rejects_invalid_candidate_k():
    service, _, _ = make_service()

    with pytest.raises(ValueError, match="candidate_k"):
        service.search("example question", candidate_k=0)


def test_hybrid_search_rejects_candidate_k_below_top_k():
    service, _, _ = make_service()

    with pytest.raises(ValueError, match="candidate_k"):
        service.search(
            "example question",
            top_k=10,
            candidate_k=5,
        )
