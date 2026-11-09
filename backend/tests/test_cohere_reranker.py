
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hybrid_rag.application.reranking_service import RerankingService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult
from hybrid_rag.infrastructure.cohere_reranker import CohereReranker


def make_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        text=f"Policy text for {chunk_id}",
        chunk_index=0,
        page_number=1,
    )


def make_client(results):
    client = Mock()
    client.rerank.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(
                index=index,
                relevance_score=score,
            )
            for index, score in results
        ]
    )
    return client


def test_maps_scores_to_original_input_order():
    client = make_client([
        (2, 0.7),
        (0, 0.2),
        (1, 0.95),
    ])

    reranker = CohereReranker(client)

    scores = reranker.rerank(
        "annual leave",
        [
            make_chunk("a"),
            make_chunk("b"),
            make_chunk("c"),
        ],
        top_k=2,
    )

    assert scores == [0.2, 0.95, 0.7]

    client.rerank.assert_called_once_with(
        model="rerank-v3.5",
        query="annual leave",
        documents=[
            "Policy text for a",
            "Policy text for b",
            "Policy text for c",
        ],
        top_n=3,
    )


def test_empty_chunks_skip_api_call():
    client = make_client([])
    reranker = CohereReranker(client)

    assert reranker.rerank("query", [], top_k=5) == []
    client.rerank.assert_not_called()


def test_empty_query_is_rejected():
    reranker = CohereReranker(make_client([]))

    with pytest.raises(ValueError, match="Query"):
        reranker.rerank(" ", [make_chunk("a")], 1)


def test_invalid_top_k_is_rejected():
    reranker = CohereReranker(make_client([]))

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("query", [make_chunk("a")], 0)


def test_missing_scores_are_rejected():
    reranker = CohereReranker(
        make_client([(0, 0.8)])
    )

    with pytest.raises(ValueError, match="every chunk"):
        reranker.rerank(
            "query",
            [make_chunk("a"), make_chunk("b")],
            2,
        )


def test_duplicate_indices_are_rejected():
    reranker = CohereReranker(
        make_client([(0, 0.8), (0, 0.7)])
    )

    with pytest.raises(ValueError, match="Duplicate"):
        reranker.rerank(
            "query",
            [make_chunk("a"), make_chunk("b")],
            2,
        )


def test_out_of_range_indices_are_rejected():
    reranker = CohereReranker(
        make_client([(3, 0.8)])
    )

    with pytest.raises(ValueError, match="invalid index"):
        reranker.rerank(
            "query",
            [make_chunk("a")],
            1,
        )


def test_nonfinite_scores_are_rejected():
    reranker = CohereReranker(
        make_client([(0, float("nan"))])
    )

    with pytest.raises(ValueError, match="relevance score"):
        reranker.rerank(
            "query",
            [make_chunk("a")],
            1,
        )


def test_reranking_service_integration():
    client = make_client([
        (1, 0.95),
        (0, 0.2),
    ])

    service = RerankingService(
        CohereReranker(client)
    )

    results = [
        SearchResult(
            chunk=make_chunk("a"),
            score=0.9,
        ),
        SearchResult(
            chunk=make_chunk("b"),
            score=0.8,
        ),
    ]

    reranked = service.rerank(
        "Which policy is most relevant?",
        results,
        top_k=2,
    )

    assert [
        result.chunk.chunk_id
        for result in reranked
    ] == ["b", "a"]

    assert reranked[0].original_rank == 2
    assert reranked[0].score == 0.95
