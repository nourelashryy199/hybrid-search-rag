
import pytest

from hybrid_rag.application.indexing_service import IndexingService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexingAction
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore
from hybrid_rag.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


class CountingFakeEmbedder:
    """Deterministic embedder that tracks embedding calls."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        self.calls += 1

        vectors = {
            "chunk-v1": (1.0, 0.0, 0.0),
            "chunk-v2": (0.0, 1.0, 0.0),
        }

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=vectors[chunk.chunk_id],
            )
            for chunk in chunks
        ]


def make_identity(version: int) -> DocumentIdentity:
    return DocumentIdentity(
        document_id="document-001",
        content_hash=("a" if version == 1 else "b") * 64,
    )


def make_chunk(version: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-v{version}",
        document_id="document-001",
        text=f"Document content version {version}",
        chunk_index=0,
        page_number=1,
    )


def create_components(tmp_path, embedder):
    repository = SQLiteDocumentRepository(
        tmp_path / "documents.db"
    )

    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="persistent_chunks",
        dimensions=3,
    )

    service = IndexingService(
        repository=repository,
        embedder=embedder,
        vector_store=store,
    )

    return service, repository, store


def test_indexed_document_survives_restart(tmp_path):
    embedder = CountingFakeEmbedder()

    # First application lifecycle.
    service, repository, store = create_components(
        tmp_path, embedder
    )

    try:
        action = service.index(
            identity=make_identity(1),
            chunks=[make_chunk(1)],
        )

        assert action == IndexingAction.CREATE
        assert embedder.calls == 1
    finally:
        store.close()

    # Simulate restart by creating new repository and Qdrant instances.
    service, repository, store = create_components(
        tmp_path, embedder
    )

    try:
        indexed = repository.get("document-001")

        assert indexed is not None
        assert indexed.identity.content_hash == "a" * 64
        assert indexed.chunk_ids == ("chunk-v1",)

        results = store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chunk-v1"

        # Identical content should not be embedded again.
        action = service.index(
            identity=make_identity(1),
            chunks=[make_chunk(1)],
        )

        assert action == IndexingAction.SKIP
        assert embedder.calls == 1
    finally:
        store.close()


def test_document_replacement_survives_restart(tmp_path):
    embedder = CountingFakeEmbedder()

    # First lifecycle: index version 1.
    service, repository, store = create_components(
        tmp_path, embedder
    )

    try:
        assert service.index(
            make_identity(1),
            [make_chunk(1)],
        ) == IndexingAction.CREATE
    finally:
        store.close()

    # Second lifecycle: replace with version 2.
    service, repository, store = create_components(
        tmp_path, embedder
    )

    try:
        assert service.index(
            make_identity(2),
            [make_chunk(2)],
        ) == IndexingAction.REPLACE
    finally:
        store.close()

    # Third lifecycle: verify version 2 remains active.
    service, repository, store = create_components(
        tmp_path, embedder
    )

    try:
        indexed = repository.get("document-001")

        assert indexed is not None
        assert indexed.identity.content_hash == "b" * 64
        assert indexed.chunk_ids == ("chunk-v2",)

        results = store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chunk-v2"

        assert service.index(
            make_identity(2),
            [make_chunk(2)],
        ) == IndexingAction.SKIP

        assert embedder.calls == 2
    finally:
        store.close()
