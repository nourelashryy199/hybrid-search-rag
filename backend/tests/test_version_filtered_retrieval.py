
from hashlib import sha256

import pytest

from hybrid_rag.application.version_filtered_retrieval import (
    VersionFilteredVectorRetriever,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.domain.retrieval import SearchResult
from hybrid_rag.infrastructure.document_repository import (
    InMemoryDocumentRepository,
)


def make_result(
    chunk_id: str,
    score: float,
    document_id: str = "doc-001",
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=f"Content for {chunk_id}",
            chunk_index=0,
            page_number=1,
        ),
        score=score,
    )


def make_indexed_document(
    document_id: str,
    chunk_ids: tuple[str, ...],
) -> IndexedDocument:
    return IndexedDocument(
        identity=DocumentIdentity(
            document_id=document_id,
            content_hash=sha256(b"version-2").hexdigest(),
        ),
        chunk_ids=chunk_ids,
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def search(self, query_vector, top_k=5):
        self.requested_k = top_k
        return self.results[:top_k]


def test_filters_stale_vectors():
    repository = InMemoryDocumentRepository()
    repository.save(
        make_indexed_document(
            "doc-001",
            ("new-chunk",),
        )
    )

    retriever = FakeRetriever([
        make_result("old-chunk", 0.95),
        make_result("new-chunk", 0.80),
    ])

    filtered = VersionFilteredVectorRetriever(
        retriever,
        repository,
        candidate_k=10,
    )

    results = filtered.search([1.0, 0.0], top_k=2)

    assert [r.chunk.chunk_id for r in results] == [
        "new-chunk"
    ]


def test_preserves_active_result_order():
    repository = InMemoryDocumentRepository()
    repository.save(
        make_indexed_document(
            "doc-001",
            ("chunk-a", "chunk-b"),
        )
    )

    retriever = FakeRetriever([
        make_result("chunk-a", 0.90),
        make_result("chunk-b", 0.70),
    ])

    filtered = VersionFilteredVectorRetriever(
        retriever,
        repository,
    )

    results = filtered.search([1.0, 0.0], top_k=2)

    assert [r.chunk.chunk_id for r in results] == [
        "chunk-a",
        "chunk-b",
    ]


def test_excludes_documents_missing_from_repository():
    repository = InMemoryDocumentRepository()

    retriever = FakeRetriever([
        make_result("orphan-chunk", 0.99),
    ])

    filtered = VersionFilteredVectorRetriever(
        retriever,
        repository,
    )

    assert filtered.search([1.0, 0.0]) == []


def test_retrieves_extra_candidates_before_filtering():
    repository = InMemoryDocumentRepository()
    repository.save(
        make_indexed_document(
            "doc-001",
            ("active-chunk",),
        )
    )

    retriever = FakeRetriever([
        make_result("stale-1", 0.99),
        make_result("stale-2", 0.95),
        make_result("active-chunk", 0.80),
    ])

    filtered = VersionFilteredVectorRetriever(
        retriever,
        repository,
        candidate_k=10,
    )

    results = filtered.search([1.0, 0.0], top_k=1)

    assert retriever.requested_k == 10
    assert results[0].chunk.chunk_id == "active-chunk"


def test_rejects_invalid_candidate_k():
    with pytest.raises(ValueError):
        VersionFilteredVectorRetriever(
            FakeRetriever([]),
            InMemoryDocumentRepository(),
            candidate_k=0,
        )


def test_rejects_invalid_top_k():
    filtered = VersionFilteredVectorRetriever(
        FakeRetriever([]),
        InMemoryDocumentRepository(),
    )

    with pytest.raises(ValueError):
        filtered.search([1.0, 0.0], top_k=0)
