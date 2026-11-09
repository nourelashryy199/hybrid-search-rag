
import pytest

from hybrid_rag.application.bm25_recovery import BM25RecoveryService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore
from hybrid_rag.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from hybrid_rag.domain.embedding import EmbeddedChunk


def make_chunk(
    chunk_id: str,
    document_id: str = "doc-001",
    text: str = "Employees receive 25 days of annual leave.",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        page_number=1,
    )


def save_active_document(
    repository,
    chunk_ids,
    document_id="doc-001",
):
    repository.save(
        IndexedDocument(
            identity=DocumentIdentity(
                document_id=document_id,
                content_hash="a" * 64,
            ),
            chunk_ids=tuple(chunk_ids),
        )
    )


def test_rebuild_bm25_after_restart(tmp_path):
    database_path = tmp_path / "documents.db"
    storage_path = tmp_path / "qdrant"

    repository = SQLiteDocumentRepository(database_path)

    store = QdrantVectorStore(
        storage_path=storage_path,
        collection_name="recovery_test",
        dimensions=3,
    )

    chunk = make_chunk("chunk-001")

    try:
        store.add(
            [chunk],
            [
                EmbeddedChunk(
                    chunk_id="chunk-001",
                    vector=(1.0, 0.0, 0.0),
                )
            ],
        )

        save_active_document(repository, ["chunk-001"])
    finally:
        store.close()

    # Simulate application restart.
    repository = SQLiteDocumentRepository(database_path)

    store = QdrantVectorStore(
        storage_path=storage_path,
        collection_name="recovery_test",
        dimensions=3,
    )

    try:
        recovered_index = BM25RecoveryService(
            repository=repository,
            chunk_reader=store,
        ).rebuild()

        results = recovered_index.search(
            "annual leave",
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk == chunk
    finally:
        store.close()


def test_rebuild_excludes_inactive_chunks(tmp_path):
    repository = SQLiteDocumentRepository(
        tmp_path / "documents.db"
    )

    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="recovery_test",
        dimensions=3,
    )

    active = make_chunk(
        "active-chunk",
        text="Annual leave is approved by managers.",
    )

    inactive = make_chunk(
        "inactive-chunk",
        text="Obsolete reimbursement instructions.",
    )

    try:
        store.add(
            [active, inactive],
            [
                EmbeddedChunk(
                    chunk_id="active-chunk",
                    vector=(1.0, 0.0, 0.0),
                ),
                EmbeddedChunk(
                    chunk_id="inactive-chunk",
                    vector=(0.0, 1.0, 0.0),
                ),
            ],
        )

        save_active_document(repository, ["active-chunk"])

        index = BM25RecoveryService(
            repository=repository,
            chunk_reader=store,
        ).rebuild()

        assert [
            result.chunk.chunk_id
            for result in index.search("annual leave", top_k=5)
        ] == ["active-chunk"]

        assert index.search(
            "reimbursement",
            top_k=5,
        ) == []
    finally:
        store.close()


def test_rebuild_rejects_missing_active_chunk(tmp_path):
    repository = SQLiteDocumentRepository(
        tmp_path / "documents.db"
    )

    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="recovery_test",
        dimensions=3,
    )

    try:
        save_active_document(repository, ["missing-chunk"])

        with pytest.raises(
            ValueError,
            match="Missing active chunks",
        ):
            BM25RecoveryService(
                repository=repository,
                chunk_reader=store,
            ).rebuild()
    finally:
        store.close()


def test_rebuild_empty_repository(tmp_path):
    repository = SQLiteDocumentRepository(
        tmp_path / "documents.db"
    )

    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="recovery_test",
        dimensions=3,
    )

    try:
        index = BM25RecoveryService(
            repository=repository,
            chunk_reader=store,
        ).rebuild()

        assert index.search("anything", top_k=5) == []
    finally:
        store.close()
