
from hashlib import sha256

from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.application.version_filtered_keyword_retrieval import (
    VersionFilteredKeywordRetriever,
)
from hybrid_rag.application.version_filtered_retrieval import (
    VersionFilteredVectorRetriever,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.infrastructure.bm25_index import BM25Index
from hybrid_rag.infrastructure.document_repository import (
    InMemoryDocumentRepository,
)
from hybrid_rag.infrastructure.vector_index import InMemoryVectorIndex
from hybrid_rag.domain.embedding import EmbeddedChunk


class FakeEmbeddingModel:
    dimensions = 3

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        chunk_index=0,
        page_number=1,
    )


def test_hybrid_retrieval_excludes_obsolete_document_chunks():
    old_chunk = make_chunk(
        "old-chunk",
        "Employees receive 10 days of annual leave.",
    )

    new_chunk = make_chunk(
        "new-chunk",
        "Employees receive 25 days of annual leave.",
    )

    chunks = [old_chunk, new_chunk]

    embeddings = [
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            vector=(1.0, 0.0, 0.0),
        )
        for chunk in chunks
    ]

    # Simulate a failed cleanup: both versions remain stored.
    vector_index = InMemoryVectorIndex(dimensions=3)
    vector_index.add(chunks, embeddings)

    bm25_index = BM25Index()
    bm25_index.add(chunks)

    # Only the new version is active.
    repository = InMemoryDocumentRepository()
    repository.save(
        IndexedDocument(
            identity=DocumentIdentity(
                document_id="doc-001",
                content_hash=sha256(b"version-2").hexdigest(),
            ),
            chunk_ids=("new-chunk",),
        )
    )

    filtered_vector = VersionFilteredVectorRetriever(
        retriever=vector_index,
        repository=repository,
        candidate_k=10,
    )

    filtered_keyword = VersionFilteredKeywordRetriever(
        retriever=bm25_index,
        repository=repository,
        candidate_k=10,
    )

    hybrid = HybridRetrievalService(
        embedding_model=FakeEmbeddingModel(),
        vector_retriever=filtered_vector,
        keyword_retriever=filtered_keyword,
        fusion=ReciprocalRankFusion(k=60),
    )

    results = hybrid.search(
        "How many annual leave days do employees receive?",
        top_k=2,
        candidate_k=2,
    )

    result_ids = [result.chunk.chunk_id for result in results]

    assert result_ids == ["new-chunk"]
    assert "old-chunk" not in result_ids
