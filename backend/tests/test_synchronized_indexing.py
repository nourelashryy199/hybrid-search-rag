
import pytest

from hybrid_rag.application.bm25_recovery import BM25RecoveryService
from hybrid_rag.application.indexing_service import IndexingService
from hybrid_rag.application.synchronized_indexing import (
    SynchronizedIndexingService,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexingAction
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore
from hybrid_rag.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


class FakeEmbedder:
    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=(1.0, 0.0, 0.0),
            )
            for chunk in chunks
        ]


def make_identity(version: int) -> DocumentIdentity:
    return DocumentIdentity(
        document_id="handbook",
        content_hash=("a" if version == 1 else "b") * 64,
    )


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="handbook",
        text=text,
        chunk_index=0,
        page_number=1,
    )


@pytest.fixture
def components(tmp_path):
    repository = SQLiteDocumentRepository(
        tmp_path / "documents.db"
    )

    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="synchronization_test",
        dimensions=3,
    )

    indexer = IndexingService(
        repository=repository,
        embedder=FakeEmbedder(),
        vector_store=store,
    )

    recovery = BM25RecoveryService(
        repository=repository,
        chunk_reader=store,
    )

    service = SynchronizedIndexingService(
        indexer=indexer,
        keyword_index_builder=recovery,
    )

    try:
        yield service, store
    finally:
        store.close()


def test_new_document_becomes_searchable_immediately(components):
    service, _ = components

    assert service.keyword_index.search(
        "annual leave",
        top_k=5,
    ) == []

    action = service.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    assert action == IndexingAction.CREATE

    results = service.keyword_index.search(
        "annual leave",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave-v1"


def test_replacement_removes_old_bm25_content(components):
    service, _ = components

    service.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    action = service.index(
        make_identity(2),
        [
            make_chunk(
                "leave-v2",
                "Employees receive 25 days of vacation.",
            )
        ],
    )

    assert action == IndexingAction.REPLACE

    assert service.keyword_index.search(
        "annual leave",
        top_k=5,
    ) == []

    results = service.keyword_index.search(
        "vacation",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave-v2"


def test_skip_does_not_replace_keyword_index(components):
    service, _ = components

    identity = make_identity(1)
    chunks = [
        make_chunk(
            "leave-v1",
            "Employees receive 20 days of annual leave.",
        )
    ]

    service.index(identity, chunks)

    original_index = service.keyword_index

    action = service.index(identity, chunks)

    assert action == IndexingAction.SKIP
    assert service.keyword_index is original_index


def test_replacement_also_updates_qdrant(components):
    service, store = components

    service.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    service.index(
        make_identity(2),
        [
            make_chunk(
                "leave-v2",
                "Employees receive 25 days of vacation.",
            )
        ],
    )

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave-v2"


def test_failed_indexing_preserves_existing_bm25(components):
    service, _ = components

    service.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    original_index = service.keyword_index

    with pytest.raises(ValueError):
        service.index(
            make_identity(2),
            [
                make_chunk(
                    "leave-v2",
                    "Employees receive 25 days of vacation.",
                ),
                make_chunk(
                    "leave-v2",
                    "Duplicate chunk ID.",
                ),
            ],
        )

    assert service.keyword_index is original_index

    results = service.keyword_index.search(
        "annual leave",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave-v1"
