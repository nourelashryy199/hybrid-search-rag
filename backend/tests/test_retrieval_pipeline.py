
import pytest

from hybrid_rag.application.retrieval_pipeline import RetrievalPipeline
from hybrid_rag.application.reranking_service import RerankingService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult


def make_result(chunk_id: str, score: float) -> SearchResult:
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


class FakeHybridRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5, candidate_k=20):
        self.calls.append((query, top_k, candidate_k))
        return self.results[:top_k]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def rerank(self, query, chunks, top_k):
        self.calls += 1
        return self.scores[:len(chunks)]


@pytest.fixture
def results():
    return [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
        make_result("chunk-c", 0.7),
        make_result("chunk-d", 0.6),
    ]


def test_pipeline_without_reranking_preserves_rrf_order(results):
    retriever = FakeHybridRetriever(results)
    pipeline = RetrievalPipeline(retriever)

    output = pipeline.search(
        "annual leave",
        top_k=2,
        candidate_k=4,
    )

    assert [
        result.chunk.chunk_id for result in output
    ] == ["chunk-a", "chunk-b"]

    assert [result.score for result in output] == [0.9, 0.8]
    assert [result.original_rank for result in output] == [1, 2]
    assert retriever.calls == [("annual leave", 2, 4)]


def test_pipeline_with_reranking_changes_order(results):
    retriever = FakeHybridRetriever(results)
    fake = FakeReranker([0.1, 0.2, 0.95, 0.5])

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=RerankingService(fake),
    )

    output = pipeline.search(
        "annual leave",
        top_k=2,
        candidate_k=4,
    )

    assert [
        result.chunk.chunk_id for result in output
    ] == ["chunk-c", "chunk-d"]

    assert [result.original_rank for result in output] == [3, 4]
    assert retriever.calls == [("annual leave", 4, 4)]
    assert fake.calls == 1


def test_pipeline_returns_empty_when_nothing_retrieved():
    retriever = FakeHybridRetriever([])
    fake = FakeReranker([])

    pipeline = RetrievalPipeline(
        retriever,
        RerankingService(fake),
    )

    assert pipeline.search("unknown query") == []
    assert fake.calls == 0


def test_pipeline_rejects_empty_query(results):
    pipeline = RetrievalPipeline(FakeHybridRetriever(results))

    with pytest.raises(ValueError, match="Query"):
        pipeline.search("   ")


def test_pipeline_rejects_invalid_top_k(results):
    pipeline = RetrievalPipeline(FakeHybridRetriever(results))

    with pytest.raises(ValueError, match="top_k"):
        pipeline.search("query", top_k=0)


def test_pipeline_rejects_candidate_k_smaller_than_top_k(results):
    pipeline = RetrievalPipeline(FakeHybridRetriever(results))

    with pytest.raises(ValueError, match="candidate_k"):
        pipeline.search("query", top_k=5, candidate_k=3)


def test_pipeline_without_reranker_does_not_expand_results(results):
    retriever = FakeHybridRetriever(results)
    pipeline = RetrievalPipeline(retriever)

    output = pipeline.search(
        "query",
        top_k=1,
        candidate_k=4,
    )

    assert len(output) == 1
    assert retriever.calls == [("query", 1, 4)]


def test_pipeline_with_reranker_considers_extra_candidates(results):
    retriever = FakeHybridRetriever(results)
    fake = FakeReranker([0.1, 0.2, 0.3, 0.99])

    pipeline = RetrievalPipeline(
        retriever,
        RerankingService(fake),
    )

    output = pipeline.search(
        "query",
        top_k=1,
        candidate_k=4,
    )

    assert len(output) == 1
    assert output[0].chunk.chunk_id == "chunk-d"
    assert output[0].original_rank == 4
