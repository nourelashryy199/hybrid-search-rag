
import pytest

from hybrid_rag.application.evaluation import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_all_relevant_retrieved():
    result = recall_at_k(
        ["a", "b", "c"],
        {"a", "b"},
        k=3,
    )

    assert result == 1.0


def test_recall_at_k_partial_retrieval():
    result = recall_at_k(
        ["a", "c", "d"],
        {"a", "b"},
        k=2,
    )

    assert result == 0.5


def test_recall_at_k_no_relevant_results():
    result = recall_at_k(
        ["c", "d"],
        {"a", "b"},
        k=2,
    )

    assert result == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(
        ["a", "b", "c"],
        {"a"},
    ) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(
        ["a", "b", "c"],
        {"c"},
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_missing_result():
    assert reciprocal_rank(
        ["a", "b"],
        {"c"},
    ) == 0.0


def test_mean_reciprocal_rank():
    result = mean_reciprocal_rank(
        rankings=[
            ["a", "b"],
            ["c", "d"],
        ],
        relevant_sets=[
            {"a"},
            {"d"},
        ],
    )

    assert result == pytest.approx(0.75)


def test_recall_rejects_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=0)


def test_evaluation_rejects_empty_relevance():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_mrr_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mean_reciprocal_rank(
            rankings=[["a"]],
            relevant_sets=[],
        )


def test_mrr_rejects_empty_dataset():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])
