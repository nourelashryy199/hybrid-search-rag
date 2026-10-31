import pytest

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.infrastructure.vector_index import (
    InMemoryVectorIndex,
    cosine_similarity,
)


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