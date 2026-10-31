
import pytest

from hybrid_rag.application.embedding_service import (
    EmbeddingService,
)
from hybrid_rag.domain.document import DocumentChunk


class FakeEmbeddingModel:
    def __init__(
        self,
        vectors: list[list[float]],
        dimensions: int = 3,
    ) -> None:
        self.vectors = vectors
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self.vectors

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.0] * self.dimensions


def make_chunk(
    chunk_id: str,
    text: str,
    index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        chunk_index=index,
        page_number=1,
    )


def test_embedding_service_preserves_chunk_ids():
    chunks = [
        make_chunk("chunk-001", "Annual leave policy", 0),
        make_chunk("chunk-002", "Remote work policy", 1),
    ]

    model = FakeEmbeddingModel(
        vectors=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    result = EmbeddingService(model).embed_chunks(chunks)

    assert len(result) == 2
    assert result[0].chunk_id == "chunk-001"
    assert result[1].chunk_id == "chunk-002"
    assert result[0].vector == (0.1, 0.2, 0.3)


def test_embedding_service_handles_empty_chunks():
    model = FakeEmbeddingModel(vectors=[])

    result = EmbeddingService(model).embed_chunks([])

    assert result == []


def test_embedding_service_rejects_wrong_vector_count():
    chunks = [
        make_chunk("chunk-001", "First text", 0),
        make_chunk("chunk-002", "Second text", 1),
    ]

    model = FakeEmbeddingModel(
        vectors=[[0.1, 0.2, 0.3]]
    )

    with pytest.raises(
        ValueError,
        match="incorrect number of vectors",
    ):
        EmbeddingService(model).embed_chunks(chunks)


def test_embedding_service_rejects_wrong_dimensions():
    chunks = [
        make_chunk("chunk-001", "Example text", 0),
    ]

    model = FakeEmbeddingModel(
        vectors=[[0.1, 0.2]],
        dimensions=3,
    )

    with pytest.raises(
        ValueError,
        match="Expected 3 dimensions",
    ):
        EmbeddingService(model).embed_chunks(chunks)


def test_embedding_service_rejects_nonfinite_values():
    chunks = [
        make_chunk("chunk-001", "Example text", 0),
    ]

    model = FakeEmbeddingModel(
        vectors=[[0.1, float("nan"), 0.3]]
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        EmbeddingService(model).embed_chunks(chunks)
