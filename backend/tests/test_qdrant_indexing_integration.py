import pytest

from hybrid_rag.application.indexing_service import IndexingService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexingAction
from hybrid_rag.infrastructure.document_repository import (
    InMemoryDocumentRepository,
)
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore


class FakeChunkEmbedder:
    """Deterministic embedding service for integration testing."""

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        vectors = {
            "old-chunk": (1.0, 0.0, 0.0),
            "new-chunk": (0.0, 1.0, 0.0),
        }

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=vectors[chunk.chunk_id],
            )
            for chunk in chunks
        ]


@pytest.fixture
def components(tmp_path):
    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="test_indexing",
        dimensions=3,
    )

    repository = InMemoryDocumentRepository()

    service = IndexingService(
        repository=repository,
        embedder=FakeChunkEmbedder(),
        vector_store=store,
    )

    try:
        yield service, repository, store
    finally:
        store.close()


def make_identity(version: str) -> DocumentIdentity:
    return DocumentIdentity(
        document_id="document-001",
        content_hash="a" * 64 if version == "v1" else "b" * 64,
    )


def make_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        text=f"Content for {chunk_id}",
        chunk_index=0,
        page_number=1,
    )


def test_indexing_creates_vectors_in_qdrant(components):
    service, repository, store = components

    action = service.index(
        identity=make_identity("v1"),
        chunks=[make_chunk("old-chunk")],
    )

    assert action == IndexingAction.CREATE

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "old-chunk"

    indexed = repository.get("document-001")

    assert indexed is not None
    assert indexed.chunk_ids == ("old-chunk",)


def test_indexing_same_version_skips_reembedding(components):
    service, repository, store = components

    identity = make_identity("v1")
    chunks = [make_chunk("old-chunk")]

    first_action = service.index(identity, chunks)
    second_action = service.index(identity, chunks)

    assert first_action == IndexingAction.CREATE
    assert second_action == IndexingAction.SKIP

    collection = store.client.get_collection(
        store.collection_name
    )

    assert collection.points_count == 1


def test_replacing_document_removes_old_qdrant_vectors(components):
    service, repository, store = components

    first_action = service.index(
        identity=make_identity("v1"),
        chunks=[make_chunk("old-chunk")],
    )

    second_action = service.index(
        identity=make_identity("v2"),
        chunks=[make_chunk("new-chunk")],
    )

    assert first_action == IndexingAction.CREATE
    assert second_action == IndexingAction.REPLACE

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "new-chunk"

    indexed = repository.get("document-001")

    assert indexed is not None
    assert indexed.identity.content_hash == "b" * 64
    assert indexed.chunk_ids == ("new-chunk",)
