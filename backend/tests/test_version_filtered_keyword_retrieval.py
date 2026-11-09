
from hashlib import sha256

import pytest

from hybrid_rag.application.version_filtered_keyword_retrieval import (
    VersionFilteredKeywordRetriever,
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


class FakeKeywordRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def search(self, query, top_k=5):
        self.requested_k = top_k
        return self.results[:top_k]


def test_filters_stale_keyword_results():
    repository = InMemoryDocumentRepository()
    repository.save(
        make_indexed_document(
            "doc-001",
            ("new-chunk",),
        )
    )

    retriever = FakeKeywordRetriever([
        make_result("old-chunk", 8.0),
        make_result("new-chunk", 5.0),
    ])

    filtered = VersionFilteredKeywordRetriever(
        retriever,
        repository,
        candidate_k=10,
    )

    results = filtered.search("annual leave", top_k=2)

    assert [r.chunk.chunk_id for r in results] == [
        "new-chunk"
    ]


def test_preserves_keyword_ranking_order():
    repository = InMemoryDocumentRepository()
    repository.save(
        make_indexed_document(
            "doc-001",
            ("chunk-a", "chunk-b"),
        )
    )

    retriever = FakeKeywordRetriever([
        make_result("chunk-a", 7.5),
        make_result("chunk-b", 4.2),
    ])

    filtered = VersionFilteredKeywordRetriever(
        retriever,
        repository,
    )

    results = filtered.search("policy", top_k=2)

    assert [r.chunk.chunk_id for r in results] == [
        "chunk-a",
        "chunk-b",
    ]


def test_excludes_unregistered_documents():
    repository = InMemoryDocumentRepository()

    retriever = FakeKeywordRetriever([
        make_result("orphan-chunk", 10.0),
    ])

    filtered = VersionFilteredKeywordRetriever(
        retriever,
        repository,
    )

    assert filtered.search("policy") == []


def test_oversamples_before_filtering():
    repository = InMemoryDocumentRepository()
    repository.save(
        make_indexed_document(
            "doc-001",
            ("active-chunk",),
        )
    )

    retriever = FakeKeywordRetriever([
        make_result("stale-1", 9.0),
        make_result("stale-2", 8.0),
        make_result("active-chunk", 7.0),
    ])

    filtered = VersionFilteredKeywordRetriever(
        retriever,
        repository,
        candidate_k=10,
    )

    results = filtered.search("policy", top_k=1)

    assert retriever.requested_k == 10
    assert results[0].chunk.chunk_id == "active-chunk"


def test_rejects_invalid_candidate_k():
    with pytest.raises(ValueError):
        VersionFilteredKeywordRetriever(
            FakeKeywordRetriever([]),
            InMemoryDocumentRepository(),
            candidate_k=0,
        )


def test_rejects_invalid_top_k():
    filtered = VersionFilteredKeywordRetriever(
        FakeKeywordRetriever([]),
        InMemoryDocumentRepository(),
    )

    with pytest.raises(ValueError):
        filtered.search("policy", top_k=0)
