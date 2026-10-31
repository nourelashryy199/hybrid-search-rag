
import pytest

from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult


def make_result(
    chunk_id: str,
    score: float = 0.5,
    text: str | None = None,
) -> SearchResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text or f"Passage {chunk_id}",
        chunk_index=0,
        page_number=1,
    )

    return SearchResult(chunk=chunk, score=score)


def test_rrf_prioritizes_results_found_by_both_retrievers():
    dense = [
        make_result("a", 0.9),
        make_result("b", 0.8),
    ]

    bm25 = [
        make_result("b", 12.0),
        make_result("c", 8.0),
    ]

    results = ReciprocalRankFusion().fuse(
        [dense, bm25]
    )

    assert results[0].chunk.chunk_id == "b"


def test_rrf_uses_ranks_not_original_scores():
    dense = [
        make_result("a", 0.1),
        make_result("b", 0.9),
    ]

    results = ReciprocalRankFusion().fuse([dense])

    assert results[0].chunk.chunk_id == "a"


def test_rrf_respects_top_k():
    rankings = [
        [
            make_result("a"),
            make_result("b"),
            make_result("c"),
        ]
    ]

    results = ReciprocalRankFusion().fuse(
        rankings,
        top_k=2,
    )

    assert len(results) == 2


def test_rrf_handles_empty_rankings():
    assert ReciprocalRankFusion().fuse([]) == []
    assert ReciprocalRankFusion().fuse([[], []]) == []


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=0)


def test_rrf_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        ReciprocalRankFusion().fuse(
            [[make_result("a")]],
            top_k=0,
        )


def test_rrf_does_not_double_count_duplicate_ids():
    ranking = [
        make_result("a"),
        make_result("a"),
        make_result("b"),
    ]

    results = ReciprocalRankFusion().fuse([ranking])

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score == pytest.approx(1 / 61)


def test_rrf_rejects_conflicting_chunk_content():
    dense = [make_result("a", text="First passage")]
    bm25 = [make_result("a", text="Different passage")]

    with pytest.raises(ValueError, match="Conflicting"):
        ReciprocalRankFusion().fuse([dense, bm25])
