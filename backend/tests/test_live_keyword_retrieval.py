
import pytest

from hybrid_rag.application.bm25_recovery import BM25RecoveryService
from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.application.indexing_service import IndexingService
from hybrid_rag.application.live_keyword_retrieval import (
    LiveKeywordRetriever,
)
from hybrid_rag.application.synchronized_indexing import (
    SynchronizedIndexingService,
)
from hybrid_rag.application.version_filtered_keyword_retrieval import (
    VersionFilteredKeywordRetriever,
)
from hybrid_rag.application.version_filtered_retrieval import (
    VersionFilteredVectorRetriever,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexingAction
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore
from hybrid_rag.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


class FakeEmbeddingModel:
    dimensions = 3

    def embed_query(self, text: str) -> list[float]:
        text = text.lower()

        if "vacation" in text:
            return [0.0, 1.0, 0.0]

        return [1.0, 0.0, 0.0]


class FakeChunkEmbedder:
    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=(
                    (0.0, 1.0, 0.0)
                    if "vacation" in chunk.text.lower()
                    else (1.0, 0.0, 0.0)
                ),
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
        collection_name="live_keyword_test",
        dimensions=3,
    )

    indexer = IndexingService(
        repository=repository,
        embedder=FakeChunkEmbedder(),
        vector_store=store,
    )

    recovery = BM25RecoveryService(
        repository=repository,
        chunk_reader=store,
    )

    synchronized = SynchronizedIndexingService(
        indexer=indexer,
        keyword_index_builder=recovery,
    )

    live_keyword = LiveKeywordRetriever(
        provider=synchronized,
    )

    keyword_retriever = VersionFilteredKeywordRetriever(
        retriever=live_keyword,
        repository=repository,
    )

    vector_retriever = VersionFilteredVectorRetriever(
        retriever=store,
        repository=repository,
    )

    retrieval = HybridRetrievalService(
        embedding_model=FakeEmbeddingModel(),
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        fusion=ReciprocalRankFusion(),
    )

    try:
        yield synchronized, live_keyword, retrieval
    finally:
        store.close()


def test_live_keyword_retriever_sees_new_documents(components):
    synchronized, live_keyword, _ = components

    assert live_keyword.search("annual leave") == []

    action = synchronized.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    assert action == IndexingAction.CREATE

    results = live_keyword.search("annual leave")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave-v1"


def test_live_keyword_retriever_sees_replacements(components):
    synchronized, live_keyword, _ = components

    synchronized.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    synchronized.index(
        make_identity(2),
        [
            make_chunk(
                "leave-v2",
                "Employees receive 25 days of vacation.",
            )
        ],
    )

    assert live_keyword.search("annual leave") == []

    results = live_keyword.search("vacation")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "leave-v2"


def test_existing_hybrid_service_sees_new_documents(components):
    synchronized, _, retrieval = components

    assert retrieval.search(
        "annual leave",
        top_k=5,
        candidate_k=10,
    ) == []

    synchronized.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    results = retrieval.search(
        "annual leave",
        top_k=5,
        candidate_k=10,
    )

    assert results
    assert results[0].chunk.chunk_id == "leave-v1"


def test_existing_hybrid_service_sees_replacements(components):
    synchronized, _, retrieval = components

    synchronized.index(
        make_identity(1),
        [
            make_chunk(
                "leave-v1",
                "Employees receive 20 days of annual leave.",
            )
        ],
    )

    synchronized.index(
        make_identity(2),
        [
            make_chunk(
                "leave-v2",
                "Employees receive 25 days of vacation.",
            )
        ],
    )

    results = retrieval.search(
        "vacation",
        top_k=5,
        candidate_k=10,
    )

    assert results
    assert results[0].chunk.chunk_id == "leave-v2"
    assert all(
        result.chunk.chunk_id != "leave-v1"
        for result in results
    )
