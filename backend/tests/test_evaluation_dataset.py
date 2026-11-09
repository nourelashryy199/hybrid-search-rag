
import pytest

from evaluation.run_retrieval_evaluation import (
    evaluate_rankings,
    load_queries,
)
from evaluation.sample_corpus import get_sample_chunks


def test_sample_corpus_has_unique_chunk_ids():
    chunks = get_sample_chunks()

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_evaluation_query_ids_are_unique():
    queries = load_queries()

    ids = [item["query_id"] for item in queries]

    assert len(ids) == len(set(ids))


def test_all_relevant_chunks_exist():
    chunks = get_sample_chunks()
    queries = load_queries()

    corpus_ids = {chunk.chunk_id for chunk in chunks}

    for item in queries:
        assert set(item["relevant_chunk_ids"]).issubset(
            corpus_ids
        )


def test_evaluation_has_twelve_queries():
    assert len(load_queries()) == 12


def test_evaluation_metrics_on_known_rankings():
    metrics = evaluate_rankings(
        rankings=[
            ["a", "b", "c"],
            ["c", "b", "a"],
        ],
        relevant_sets=[
            {"a"},
            {"b"},
        ],
    )

    assert metrics["Recall@1"] == pytest.approx(0.5)
    assert metrics["Recall@3"] == pytest.approx(1.0)
    assert metrics["MRR"] == pytest.approx(0.75)
