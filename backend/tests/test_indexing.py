
import pytest

from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import (
    IndexedDocument,
    IndexingAction,
    determine_indexing_action,
)


def make_identity(
    document_id: str = "policy-001",
    content_hash: str = "a" * 64,
) -> DocumentIdentity:
    return DocumentIdentity(
        document_id=document_id,
        content_hash=content_hash,
    )


def test_new_document_requires_creation():
    incoming = make_identity()

    action = determine_indexing_action(
        incoming=incoming,
        existing=None,
    )

    assert action == IndexingAction.CREATE


def test_unchanged_document_is_skipped():
    identity = make_identity()

    existing = IndexedDocument(
        identity=identity,
        chunk_ids=("chunk-001", "chunk-002"),
    )

    action = determine_indexing_action(
        incoming=identity,
        existing=existing,
    )

    assert action == IndexingAction.SKIP


def test_modified_document_requires_replacement():
    existing = IndexedDocument(
        identity=make_identity(
            content_hash="a" * 64,
        ),
        chunk_ids=("old-chunk-001",),
    )

    incoming = make_identity(
        content_hash="b" * 64,
    )

    action = determine_indexing_action(
        incoming=incoming,
        existing=existing,
    )

    assert action == IndexingAction.REPLACE


def test_different_document_ids_cannot_be_compared():
    existing = IndexedDocument(
        identity=make_identity(
            document_id="policy-001",
        ),
        chunk_ids=("chunk-001",),
    )

    incoming = make_identity(
        document_id="policy-002",
    )

    with pytest.raises(
        ValueError,
        match="different logical documents",
    ):
        determine_indexing_action(
            incoming=incoming,
            existing=existing,
        )


def test_indexed_document_preserves_chunk_ids():
    indexed = IndexedDocument(
        identity=make_identity(),
        chunk_ids=("chunk-001", "chunk-002"),
    )

    assert indexed.chunk_ids == (
        "chunk-001",
        "chunk-002",
    )
