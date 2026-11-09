

from hashlib import sha256

from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.infrastructure.document_repository import (
    InMemoryDocumentRepository,
)


def make_document(
    document_id: str = "doc-001",
    content_hash: str = "hash-v1",
) -> IndexedDocument:
    valid_hash = sha256(
        content_hash.encode("utf-8")
    ).hexdigest()

    return IndexedDocument(
        identity=DocumentIdentity(
            document_id=document_id,
            content_hash=valid_hash,
        ),
        chunk_ids=("chunk-001", "chunk-002"),
    )


def test_repository_starts_empty():
    repository = InMemoryDocumentRepository()

    assert repository.get("doc-001") is None
    assert repository.list_all() == []


def test_repository_saves_document():
    repository = InMemoryDocumentRepository()
    document = make_document()

    repository.save(document)

    assert repository.get("doc-001") == document


def test_repository_replaces_existing_document():
    repository = InMemoryDocumentRepository()

    original = make_document(content_hash="hash-v1")
    updated = make_document(content_hash="hash-v2")

    repository.save(original)
    repository.save(updated)

    assert repository.get("doc-001") == updated
    assert len(repository.list_all()) == 1


def test_repository_deletes_document():
    repository = InMemoryDocumentRepository()
    repository.save(make_document())

    repository.delete("doc-001")

    assert repository.get("doc-001") is None


def test_repository_delete_missing_document():
    repository = InMemoryDocumentRepository()

    repository.delete("missing")

    assert repository.list_all() == []


def test_repository_lists_multiple_documents():
    repository = InMemoryDocumentRepository()

    repository.save(make_document("doc-001"))
    repository.save(make_document("doc-002"))

    assert len(repository.list_all()) == 2
