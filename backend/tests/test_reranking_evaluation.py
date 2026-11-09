
import pytest

from hybrid_rag.application.reranking_evaluation import (
    EvaluationQuery,
    compare_retrieval_systems,
    evaluate_retrieval,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.reranking import RerankedResult


def make_result(chunk_id: str, rank: int) -> RerankedResult:
    return RerankedResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            text=f"Content for {chunk_id}",
            chunk_index=rank - 1,
            page_number=1,
        ),
        score=1.0 / rank,
        original_rank=rank,
    )


class FakeRetrievalSystem:
    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, query, top_k=5, candidate_k=20):
        chunk_ids = self.rankings.get(query, [])

        return [
            make_result(chunk_id, rank)
            for rank, chunk_id in enumerate(
                chunk_ids[:top_k],
                start=1,
            )
        ]


@pytest.fixture
def queries():
    return [
        EvaluationQuery(
            query="annual leave",
            relevant_chunk_ids=frozenset({"leave"}),
        ),
        EvaluationQuery(
            query="authentication",
            relevant_chunk_ids=frozenset({"security"}),
        ),
        EvaluationQuery(
            query="expenses",
            relevant_chunk_ids=frozenset({"finance"}),
        ),
    ]


def test_perfect_retrieval_scores_one(queries):
    system = FakeRetrievalSystem({
        "annual leave": ["leave", "other"],
        "authentication": ["security", "other"],
        "expenses": ["finance", "other"],
    })

    metrics = evaluate_retrieval(system, queries)

    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.mrr == 1.0


def test_metrics_reflect_relevant_rank(queries):
    system = FakeRetrievalSystem({
        "annual leave": ["other", "leave"],
        "authentication": ["security", "other"],
        "expenses": ["other", "other-2", "finance"],
    })

    metrics = evaluate_retrieval(system, queries)

    assert metrics.recall_at_1 == pytest.approx(1 / 3)
    assert metrics.recall_at_3 == 1.0
    assert metrics.mrr == pytest.approx(
        (1 / 2 + 1 + 1 / 3) / 3
    )


def test_missing_relevant_chunk_scores_zero():
    system = FakeRetrievalSystem({
        "unknown": ["a", "b", "c"],
    })

    metrics = evaluate_retrieval(
        system,
        [
            EvaluationQuery(
                query="unknown",
                relevant_chunk_ids=frozenset({"missing"}),
            )
        ],
    )

    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 0.0
    assert metrics.mrr == 0.0


def test_comparison_calculates_improvement(queries):
    baseline = FakeRetrievalSystem({
        "annual leave": ["other", "leave"],
        "authentication": ["other", "security"],
        "expenses": ["other", "finance"],
    })

    reranked = FakeRetrievalSystem({
        "annual leave": ["leave", "other"],
        "authentication": ["security", "other"],
        "expenses": ["finance", "other"],
    })

    comparison = compare_retrieval_systems(
        baseline,
        reranked,
        queries,
    )

    assert comparison.baseline.recall_at_1 == 0.0
    assert comparison.reranked.recall_at_1 == 1.0

    assert comparison.recall_at_1_delta == 1.0
    assert comparison.recall_at_3_delta == 0.0
    assert comparison.mrr_delta == pytest.approx(0.5)


def test_empty_evaluation_set_is_rejected():
    system = FakeRetrievalSystem({})

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_retrieval(system, [])


def test_invalid_candidate_k_is_rejected(queries):
    system = FakeRetrievalSystem({})

    with pytest.raises(ValueError, match="candidate_k"):
        evaluate_retrieval(
            system,
            queries,
            candidate_k=2,
        )


def test_query_without_relevance_labels_is_rejected():
    system = FakeRetrievalSystem({})

    with pytest.raises(ValueError, match="relevant chunk IDs"):
        evaluate_retrieval(
            system,
            [
                EvaluationQuery(
                    query="test",
                    relevant_chunk_ids=frozenset(),
                )
            ],
        )


def test_duplicate_results_are_rejected():
    system = FakeRetrievalSystem({
        "test": ["a", "a", "b"],
    })

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_retrieval(
            system,
            [
                EvaluationQuery(
                    query="test",
                    relevant_chunk_ids=frozenset({"a"}),
                )
            ],
        )


def test_multiple_relevant_chunks_use_hit_based_recall():
    system = FakeRetrievalSystem({
        "test": ["irrelevant", "relevant-b", "other"],
    })

    metrics = evaluate_retrieval(
        system,
        [
            EvaluationQuery(
                query="test",
                relevant_chunk_ids=frozenset({
                    "relevant-a",
                    "relevant-b",
                }),
            )
        ],
    )

    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.mrr == pytest.approx(0.5)
