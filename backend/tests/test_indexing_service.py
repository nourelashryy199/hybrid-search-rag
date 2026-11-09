
from hashlib import sha256

import pytest

from hybrid_rag.application.indexing_service import IndexingService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexingAction
from hybrid_rag.infrastructure.document_repository import (
    InMemoryDocumentRepository,
)


def make_identity(
    version: str = "v1",
) -> DocumentIdentity:
    return DocumentIdentity(
        document_id="doc-001",
        content_hash=sha256(version.encode()).hexdigest(),
    )


def make_chunk(
    chunk_id: str = "chunk-001",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=f"Content for {chunk_id}",
        chunk_index=0,
        page_number=1,
    )


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_chunks(self, chunks):
        self.calls += 1
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=(1.0, 0.0, 0.0),
            )
            for chunk in chunks
        ]


class FakeVectorStore:
    def __init__(self):
        self.entries = {}
        self.add_calls = 0
        self.delete_calls = []

    def add(self, chunks, embeddings):
        self.add_calls += 1
        for chunk, embedding in zip(chunks, embeddings):
            self.entries[chunk.chunk_id] = embedding.vector

    def delete(self, chunk_ids):
        self.delete_calls.append(chunk_ids)
        for chunk_id in chunk_ids:
            self.entries.pop(chunk_id, None)


def make_service():
    repository = InMemoryDocumentRepository()
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    service = IndexingService(
        repository=repository,
        embedder=embedder,
        vector_store=store,
    )

    return service, repository, embedder, store


def test_create_new_document():
    service, repository, embedder, store = make_service()

    action = service.index(
        make_identity(),
        [make_chunk()],
    )

    assert action == IndexingAction.CREATE
    assert repository.get("doc-001") is not None
    assert "chunk-001" in store.entries
    assert embedder.calls == 1


def test_skip_unchanged_document():
    service, repository, embedder, store = make_service()

    identity = make_identity()
    chunks = [make_chunk()]

    service.index(identity, chunks)
    action = service.index(identity, chunks)

    assert action == IndexingAction.SKIP
    assert embedder.calls == 1
    assert store.add_calls == 1


def test_replace_changed_document():
    service, repository, embedder, store = make_service()

    service.index(
        make_identity("v1"),
        [make_chunk("old-chunk")],
    )

    action = service.index(
        make_identity("v2"),
        [make_chunk("new-chunk")],
    )

    assert action == IndexingAction.REPLACE
    assert "old-chunk" not in store.entries
    assert "new-chunk" in store.entries
    assert repository.get("doc-001").identity == make_identity("v2")


def test_reject_empty_chunks():
    service, _, _, _ = make_service()

    with pytest.raises(ValueError):
        service.index(make_identity(), [])


def test_reject_mismatched_document_ids():
    service, _, _, _ = make_service()

    chunk = make_chunk().model_copy(
        update={"document_id": "different-doc"}
    )

    with pytest.raises(ValueError):
        service.index(make_identity(), [chunk])


def test_reject_duplicate_chunk_ids():
    service, _, _, _ = make_service()

    with pytest.raises(ValueError):
        service.index(
            make_identity(),
            [make_chunk(), make_chunk()],
        )


def test_embedding_failure_preserves_existing_index():
    service, repository, embedder, store = make_service()

    service.index(
        make_identity("v1"),
        [make_chunk("old-chunk")],
    )

    def fail(_chunks):
        raise RuntimeError("Embedding failed")

    embedder.embed_chunks = fail

    with pytest.raises(RuntimeError):
        service.index(
            make_identity("v2"),
            [make_chunk("new-chunk")],
        )

    assert "old-chunk" in store.entries
    assert "new-chunk" not in store.entries
    assert repository.get("doc-001").identity == make_identity("v1")


def test_replacement_rejects_reused_chunk_ids():
    service, repository, embedder, store = make_service()

    service.index(
        make_identity("v1"),
        [make_chunk("shared-chunk")],
    )

    with pytest.raises(ValueError, match="version-specific IDs"):
        service.index(
            make_identity("v2"),
            [make_chunk("shared-chunk")],
        )

    assert "shared-chunk" in store.entries
    assert store.add_calls == 1
    assert store.delete_calls == []
    assert repository.get("doc-001").identity == make_identity("v1")



def test_failed_old_vector_deletion_preserves_new_version():
    service, repository, embedder, store = make_service()

    service.index(
        make_identity("v1"),
        [make_chunk("old-chunk")],
    )

    original_delete = store.delete

    def fail_old_deletion(chunk_ids):
        if chunk_ids == ["old-chunk"]:
            raise RuntimeError("Vector deletion failed")
        return original_delete(chunk_ids)

    store.delete = fail_old_deletion

    action = service.index(
        make_identity("v2"),
        [make_chunk("new-chunk")],
    )

    assert action == IndexingAction.REPLACE
    assert repository.get("doc-001").identity == make_identity("v2")
    assert "new-chunk" in store.entries

    # Cleanup failed, so the obsolete vector remains temporarily.
    assert "old-chunk" in store.entries


def test_metadata_failure_preserves_previous_version():
    service, repository, embedder, store = make_service()

    service.index(
        make_identity("v1"),
        [make_chunk("old-chunk")],
    )

    def fail_save(_document):
        raise RuntimeError("Metadata persistence failed")

    repository.save = fail_save

    with pytest.raises(
        RuntimeError,
        match="Metadata persistence failed",
    ):
        service.index(
            make_identity("v2"),
            [make_chunk("new-chunk")],
        )

    assert repository.get("doc-001").identity == make_identity("v1")
    assert "old-chunk" in store.entries
    assert "new-chunk" not in store.entries
