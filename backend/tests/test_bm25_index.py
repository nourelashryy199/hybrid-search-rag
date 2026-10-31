import pytest

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.infrastructure.bm25_index import (
    BM25Index,
    tokenize,
)


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        chunk_index=0,
        page_number=1,
    )


def test_tokenization():
    assert tokenize("Annual Leave, 25 Days!") == [
        "annual", "leave", "25", "days"
    ]


def test_empty_index_returns_no_results():
    index = BM25Index()

    assert index.search("annual leave") == []


def test_bm25_returns_matching_passage():
    index = BM25Index()

    index.add([
        make_chunk("leave", "Employees receive annual leave."),
        make_chunk("remote", "Remote work requires approval."),
        make_chunk("security", "Passwords must remain secure."),
        make_chunk("expenses", "Travel expenses require receipts."),
    ])

    results = index.search("annual leave")

    assert results
    assert results[0].chunk.chunk_id == "leave"


def test_bm25_respects_top_k():
    index = BM25Index()

    index.add([
        make_chunk("a", "Annual leave policy."),
        make_chunk("b", "Annual leave entitlement."),
        make_chunk("c", "Remote work policy."),
        make_chunk("d", "Security requirements."),
    ])

    results = index.search("annual", top_k=1)

    assert len(results) == 1


def test_bm25_rejects_invalid_top_k():
    index = BM25Index()

    with pytest.raises(ValueError):
        index.search("annual leave", top_k=0)


def test_bm25_handles_blank_query():
    index = BM25Index()

    index.add([
        make_chunk("leave", "Annual leave policy."),
    ])

    assert index.search("   ") == []


def test_bm25_replaces_existing_chunk():
    index = BM25Index()

    index.add([
        make_chunk("chunk-001", "Annual leave policy."),
    ])

    index.add([
        make_chunk("chunk-001", "Remote work policy."),
    ])

    assert index.search("annual") == []