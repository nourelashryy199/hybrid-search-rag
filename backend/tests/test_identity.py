
import pytest

from hybrid_rag.domain.identity import (
    DocumentIdentity,
    calculate_file_hash,
    create_document_identity,
)


def test_same_file_produces_same_hash(tmp_path):
    path = tmp_path / "policy.pdf"
    path.write_bytes(b"Example PDF content")

    first = calculate_file_hash(path)
    second = calculate_file_hash(path)

    assert first == second
    assert len(first) == 64


def test_changed_file_produces_different_hash(tmp_path):
    path = tmp_path / "policy.pdf"
    path.write_bytes(b"Version one")

    first = calculate_file_hash(path)

    path.write_bytes(b"Version two")

    second = calculate_file_hash(path)

    assert first != second


def test_identity_preserves_logical_document_id(tmp_path):
    path = tmp_path / "policy.pdf"
    path.write_bytes(b"Policy contents")

    identity = create_document_identity(
        path,
        document_id="policy-001",
    )

    assert identity.document_id == "policy-001"
    assert len(identity.content_hash) == 64


def test_identity_rejects_empty_document_id():
    with pytest.raises(ValueError):
        DocumentIdentity(
            document_id="",
            content_hash="a" * 64,
        )


def test_identity_rejects_invalid_hash():
    with pytest.raises(ValueError):
        DocumentIdentity(
            document_id="policy-001",
            content_hash="not-a-valid-hash",
        )


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        calculate_file_hash(
            tmp_path / "missing.pdf"
        )
