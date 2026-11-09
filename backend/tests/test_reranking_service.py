
import pytest

from hybrid_rag.application.reranking_service import (
    RerankingService,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int,
    ) -> list[float]:
        self.calls += 1
        return self.scores


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            text=f"Content for {chunk_id}",
            chunk_index=0,
            page_number=1,
        ),
        score=score,
    )


def test_reranker_changes_result_order():
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
        make_result("chunk-c", 0.7),
    ]

    service = RerankingService(
        FakeReranker([0.2, 0.95, 0.5])
    )

    reranked = service.rerank(
        "Which chunk is most relevant?",
        results,
        top_k=3,
    )

    assert [
        result.chunk.chunk_id
        for result in reranked
    ] == ["chunk-b", "chunk-c", "chunk-a"]


def test_reranker_preserves_original_rank():
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
    ]

    service = RerankingService(
        FakeReranker([0.2, 0.95])
    )

    reranked = service.rerank("query", results)

    assert reranked[0].original_rank == 2
    assert reranked[1].original_rank == 1


def test_reranker_respects_top_k():
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
        make_result("chunk-c", 0.7),
    ]

    service = RerankingService(
        FakeReranker([0.1, 0.9, 0.5])
    )

    reranked = service.rerank(
        "query",
        results,
        top_k=2,
    )

    assert len(reranked) == 2
    assert [
        result.chunk.chunk_id
        for result in reranked
    ] == ["chunk-b", "chunk-c"]


def test_equal_scores_preserve_original_order():
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
    ]

    service = RerankingService(
        FakeReranker([0.5, 0.5])
    )

    reranked = service.rerank("query", results)

    assert [
        result.chunk.chunk_id
        for result in reranked
    ] == ["chunk-a", "chunk-b"]


def test_empty_results_do_not_call_reranker():
    fake = FakeReranker([])
    service = RerankingService(fake)

    assert service.rerank("query", []) == []
    assert fake.calls == 0


def test_empty_query_is_rejected():
    service = RerankingService(FakeReranker([]))

    with pytest.raises(ValueError, match="Query"):
        service.rerank("   ", [])


def test_invalid_top_k_is_rejected():
    service = RerankingService(FakeReranker([]))

    with pytest.raises(ValueError, match="top_k"):
        service.rerank("query", [], top_k=0)


def test_mismatched_score_count_is_rejected():
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
    ]

    service = RerankingService(
        FakeReranker([0.5])
    )

    with pytest.raises(ValueError, match="one score"):
        service.rerank("query", results)


def test_nonfinite_scores_are_rejected():
    results = [make_result("chunk-a", 0.9)]

    service = RerankingService(
        FakeReranker([float("nan")])
    )

    with pytest.raises(ValueError, match="non-finite"):
        service.rerank("query", results)


def test_duplicate_chunk_ids_are_rejected():
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-a", 0.8),
    ]

    service = RerankingService(
        FakeReranker([0.2, 0.9])
    )

    with pytest.raises(ValueError, match="Duplicate"):
        service.rerank("query", results)
