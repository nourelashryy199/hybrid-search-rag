import pytest

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.infrastructure.vector_index import (
    InMemoryVectorIndex,
    cosine_similarity,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.infrastructure.vector_index import InMemoryVectorIndex


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        chunk_index=0,
        page_number=1,
    )


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity(
        (1.0, 0.0),
        (1.0, 0.0),
    ) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity(
        (1.0, 0.0),
        (0.0, 1.0),
    ) == pytest.approx(0.0)


def test_cosine_similarity_rejects_zero_vectors():
    with pytest.raises(ValueError):
        cosine_similarity(
            (0.0, 0.0),
            (1.0, 0.0),
        )


def test_vector_index_returns_best_match():
    index = InMemoryVectorIndex(dimensions=2)

    chunks = [
        make_chunk("leave", "Annual leave policy"),
        make_chunk("remote", "Remote work policy"),
    ]

    embeddings = [
        EmbeddedChunk(
            chunk_id="leave",
            vector=(1.0, 0.0),
        ),
        EmbeddedChunk(
            chunk_id="remote",
            vector=(0.0, 1.0),
        ),
    ]

    index.add(chunks, embeddings)

    results = index.search([0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave"


def test_vector_index_respects_top_k():
    index = InMemoryVectorIndex(dimensions=2)

    chunks = [
        make_chunk("a", "First"),
        make_chunk("b", "Second"),
        make_chunk("c", "Third"),
    ]

    embeddings = [
        EmbeddedChunk(chunk_id="a", vector=(1.0, 0.0)),
        EmbeddedChunk(chunk_id="b", vector=(0.8, 0.2)),
        EmbeddedChunk(chunk_id="c", vector=(0.0, 1.0)),
    ]

    index.add(chunks, embeddings)

    results = index.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]


def test_vector_index_rejects_mismatched_ids():
    index = InMemoryVectorIndex(dimensions=2)

    with pytest.raises(ValueError, match="IDs do not match"):
        index.add(
            [make_chunk("chunk-001", "Example")],
            [
                EmbeddedChunk(
                    chunk_id="chunk-002",
                    vector=(1.0, 0.0),
                )
            ],
        )


def test_vector_index_rejects_wrong_dimensions():
    index = InMemoryVectorIndex(dimensions=2)

    with pytest.raises(ValueError, match="dimensions"):
        index.add(
            [make_chunk("chunk-001", "Example")],
            [
                EmbeddedChunk(
                    chunk_id="chunk-001",
                    vector=(1.0, 0.0, 0.0),
                )
            ],
        )


def test_vector_index_empty_search():
    index = InMemoryVectorIndex(dimensions=2)

    assert index.search([1.0, 0.0]) == []


def test_vector_index_rejects_zero_query():
    index = InMemoryVectorIndex(dimensions=2)

    with pytest.raises(ValueError, match="cannot be zero"):
        index.search([0.0, 0.0])


def test_vector_index_rejects_invalid_top_k():
    index = InMemoryVectorIndex(dimensions=2)

    with pytest.raises(ValueError, match="top_k"):
        index.search([1.0, 0.0], top_k=0)


def test_delete_removes_existing_vector():
    index = InMemoryVectorIndex(dimensions=3)

    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Annual leave policy",
        chunk_index=0,
        page_number=1,
    )

    embedding = EmbeddedChunk(
        chunk_id="chunk-001",
        vector=(1.0, 0.0, 0.0),
    )

    index.add([chunk], [embedding])
    index.delete(["chunk-001"])

    assert index.search([1.0, 0.0, 0.0]) == []


def test_delete_preserves_unrelated_vectors():
    index = InMemoryVectorIndex(dimensions=3)

    first = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Annual leave policy",
        chunk_index=0,
        page_number=1,
    )

    second = DocumentChunk(
        chunk_id="chunk-002",
        document_id="doc-002",
        text="Security policy",
        chunk_index=0,
        page_number=1,
    )

    embeddings = [
        EmbeddedChunk(
            chunk_id="chunk-001",
            vector=(1.0, 0.0, 0.0),
        ),
        EmbeddedChunk(
            chunk_id="chunk-002",
            vector=(0.0, 1.0, 0.0),
        ),
    ]

    index.add([first, second], embeddings)
    index.delete(["chunk-001"])

    results = index.search([0.0, 1.0, 0.0])

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-002"


def test_delete_missing_vector_is_safe():
    index = InMemoryVectorIndex(dimensions=3)

    index.delete(["missing-chunk"])

    assert index.search([1.0, 0.0, 0.0]) == []


def test_delete_multiple_vectors():
    index = InMemoryVectorIndex(dimensions=3)

    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{i}",
            document_id="doc-001",
            text=f"Policy {i}",
            chunk_index=i,
            page_number=1,
        )
        for i in range(3)
    ]

    embeddings = [
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            vector=(1.0, 0.0, 0.0),
        )
        for chunk in chunks
    ]

    index.add(chunks, embeddings)

    index.delete(["chunk-0", "chunk-1"])

    results = index.search([1.0, 0.0, 0.0])

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-2"
