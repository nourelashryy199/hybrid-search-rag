import pytest

from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


@pytest.fixture
def repository(tmp_path):
    return SQLiteDocumentRepository(
        tmp_path / "metadata" / "documents.db"
    )


def make_document(
    document_id: str = "doc-001",
    version: str = "v1",
    chunk_ids: tuple[str, ...] = ("chunk-001", "chunk-002"),
) -> IndexedDocument:
    return IndexedDocument(
        identity=DocumentIdentity(
            document_id=document_id,
            content_hash="a" * 64 if version == "v1" else "b" * 64,
        ),
        chunk_ids=chunk_ids,
    )


def test_get_missing_document_returns_none(repository):
    assert repository.get("missing") is None


def test_save_and_get_document(repository):
    document = make_document()

    repository.save(document)

    assert repository.get("doc-001") == document


def test_save_replaces_document_version(repository):
    old = make_document()
    new = make_document(
        version="v2",
        chunk_ids=("chunk-003",),
    )

    repository.save(old)
    repository.save(new)

    assert repository.get("doc-001") == new


def test_save_preserves_chunk_order(repository):
    document = make_document(
        chunk_ids=("chunk-c", "chunk-a", "chunk-b"),
    )

    repository.save(document)

    assert repository.get("doc-001").chunk_ids == (
        "chunk-c",
        "chunk-a",
        "chunk-b",
    )


def test_delete_removes_document(repository):
    repository.save(make_document())

    repository.delete("doc-001")

    assert repository.get("doc-001") is None


def test_delete_missing_document_is_harmless(repository):
    repository.delete("missing")

    assert repository.list_all() == []


def test_list_all_returns_documents(repository):
    first = make_document(
        document_id="doc-001",
        chunk_ids=("chunk-001",),
    )
    second = make_document(
        document_id="doc-002",
        chunk_ids=("chunk-002",),
    )

    repository.save(first)
    repository.save(second)

    assert repository.list_all() == [first, second]


def test_repository_persists_across_instances(tmp_path):
    database_path = tmp_path / "documents.db"
    document = make_document()

    first_repository = SQLiteDocumentRepository(database_path)
    first_repository.save(document)

    second_repository = SQLiteDocumentRepository(database_path)

    assert second_repository.get("doc-001") == document


def test_saving_same_document_twice_is_idempotent(repository):
    document = make_document()

    repository.save(document)
    repository.save(document)

    assert repository.get("doc-001") == document
    assert len(repository.list_all()) == 1


def test_save_rolls_back_if_chunk_insertion_fails(repository):
    original = make_document(
        chunk_ids=("original-chunk",),
    )
    repository.save(original)

    invalid_replacement = make_document(
        version="v2",
        chunk_ids=("duplicate", "duplicate"),
    )

    with pytest.raises(Exception):
        repository.save(invalid_replacement)

    assert repository.get("doc-001") == original
