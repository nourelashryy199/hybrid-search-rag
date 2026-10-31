
import pytest
from pydantic import ValidationError

from hybrid_rag.domain.document import (
    Document,
    DocumentChunk,
    generate_chunk_id,
)


def test_document_creation():
    document = Document(
        document_id="doc-001",
        filename="policy.pdf",
        content="Employees are entitled to annual leave.",
        source_type="pdf",
    )

    assert document.document_id == "doc-001"
    assert document.source_type == "pdf"


def test_document_rejects_empty_content():
    with pytest.raises(ValidationError):
        Document(
            document_id="doc-001",
            filename="policy.pdf",
            content="",
            source_type="pdf",
        )


def test_chunk_creation():
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Employees are entitled to annual leave.",
        chunk_index=0,
        page_number=1,
    )

    assert chunk.chunk_index == 0
    assert chunk.page_number == 1


def test_chunk_rejects_negative_index():
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            text="Some text",
            chunk_index=-1,
        )


def test_chunk_id_is_deterministic():
    first = generate_chunk_id(
        "doc-001", 0, "Some document text"
    )
    second = generate_chunk_id(
        "doc-001", 0, "Some document text"
    )

    assert first == second
    assert len(first) == 64


def test_chunk_id_changes_with_content():
    first = generate_chunk_id("doc-001", 0, "Text A")
    second = generate_chunk_id("doc-001", 0, "Text B")

    assert first != second


def test_chunk_id_rejects_invalid_index():
    with pytest.raises(ValueError):
        generate_chunk_id("doc-001", -1, "Some text")
