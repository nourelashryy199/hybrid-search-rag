
from hybrid_rag.application.bm25_recovery import BM25RecoveryService
from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.application.indexing_service import IndexingService
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
    """Deterministic embeddings for repeatable integration tests."""

    dimensions = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            list(self._vector_for_text(text))
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector_for_text(text))

    @staticmethod
    def _vector_for_text(text: str) -> tuple[float, ...]:
        text = text.lower()

        if "leave" in text or "vacation" in text:
            return (1.0, 0.0, 0.0)

        if "authentication" in text or "password" in text:
            return (0.0, 1.0, 0.0)

        return (0.0, 0.0, 1.0)


class FakeChunkEmbedder:
    def __init__(self, model: FakeEmbeddingModel) -> None:
        self.model = model

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        vectors = self.model.embed_documents(
            [chunk.text for chunk in chunks]
        )

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=tuple(vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        page_number=1,
    )


def make_identity(document_id: str, version: int) -> DocumentIdentity:
    return DocumentIdentity(
        document_id=document_id,
        content_hash=("a" if version == 1 else "b") * 64,
    )


def create_storage(tmp_path):
    repository = SQLiteDocumentRepository(
        tmp_path / "documents.db"
    )

    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="hybrid_test",
        dimensions=3,
    )

    return repository, store


def create_retrieval_service(repository, store, model):
    bm25_index = BM25RecoveryService(
        repository=repository,
        chunk_reader=store,
    ).rebuild()

    vector_retriever = VersionFilteredVectorRetriever(
        retriever=store,
        repository=repository,
    )

    keyword_retriever = VersionFilteredKeywordRetriever(
        retriever=bm25_index,
        repository=repository,
    )

    return HybridRetrievalService(
        embedding_model=model,
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        fusion=ReciprocalRankFusion(),
    )


def test_hybrid_retrieval_survives_restart(tmp_path):
    model = FakeEmbeddingModel()

    # Lifecycle 1: index two documents.
    repository, store = create_storage(tmp_path)

    try:
        service = IndexingService(
            repository=repository,
            embedder=FakeChunkEmbedder(model),
            vector_store=store,
        )

        assert service.index(
            make_identity("handbook", 1),
            [
                make_chunk(
                    "leave-001",
                    "handbook",
                    "Employees receive 25 days of annual leave.",
                )
            ],
        ) == IndexingAction.CREATE

        assert service.index(
            make_identity("security-policy", 1),
            [
                make_chunk(
                    "security-001",
                    "security-policy",
                    "Multi-factor authentication is required.",
                )
            ],
        ) == IndexingAction.CREATE
    finally:
        store.close()

    # Lifecycle 2: reconstruct the retrieval pipeline.
    repository, store = create_storage(tmp_path)

    try:
        retrieval = create_retrieval_service(
            repository, store, model
        )

        leave_results = retrieval.search(
            "How many days of annual leave do employees get?",
            top_k=2,
            candidate_k=5,
        )

        assert leave_results
        assert leave_results[0].chunk.chunk_id == "leave-001"

        security_results = retrieval.search(
            "Is authentication required?",
            top_k=2,
            candidate_k=5,
        )

        assert security_results
        assert security_results[0].chunk.chunk_id == "security-001"
    finally:
        store.close()


def test_retrieval_excludes_inactive_document_version(tmp_path):
    model = FakeEmbeddingModel()
    repository, store = create_storage(tmp_path)

    try:
        indexing = IndexingService(
            repository=repository,
            embedder=FakeChunkEmbedder(model),
            vector_store=store,
        )

        indexing.index(
            make_identity("handbook", 1),
            [
                make_chunk(
                    "leave-old",
                    "handbook",
                    "Employees receive 20 days of annual leave.",
                )
            ],
        )

        indexing.index(
            make_identity("handbook", 2),
            [
                make_chunk(
                    "leave-new",
                    "handbook",
                    "Employees receive 25 days of annual leave.",
                )
            ],
        )

        retrieval = create_retrieval_service(
            repository, store, model
        )

        results = retrieval.search(
            "How many days of annual leave do employees get?",
            top_k=5,
            candidate_k=10,
        )

        assert results
        assert results[0].chunk.chunk_id == "leave-new"
        assert all(
            result.chunk.chunk_id != "leave-old"
            for result in results
        )
    finally:
        store.close()
