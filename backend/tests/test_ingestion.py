
from pathlib import Path

import pytest

from hybrid_rag.application.chunking import (
    ChunkingConfig,
    TextChunker,
)
from hybrid_rag.application.ingestion import (
    DocumentIngestionService,
)
from hybrid_rag.domain.document import DocumentPage


class FakeParser:
    def __init__(
        self,
        pages: list[DocumentPage],
    ) -> None:
        self.pages = pages

    def parse(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> list[DocumentPage]:
        return self.pages


def make_service(
    pages: list[DocumentPage],
) -> DocumentIngestionService:
    return DocumentIngestionService(
        parser=FakeParser(pages),
        chunker=TextChunker(
            ChunkingConfig(
                chunk_size=100,
                chunk_overlap=20,
            )
        ),
    )


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "policy.pdf"
    path.write_bytes(b"Example document contents")
    return path


def test_ingestion_returns_chunks(sample_file):
    pages = [
        DocumentPage(
            document_id="doc-001",
            page_number=1,
            text="Employees receive annual leave.",
        )
    ]

    service = make_service(pages)

    result = service.ingest(
        sample_file,
        document_id="doc-001",
    )

    assert result.document_id == "doc-001"
    assert result.filename == "policy.pdf"
    assert result.chunk_count == 1
    assert result.chunks[0].page_number == 1
    assert len(result.content_hash) == 64


def test_ingestion_preserves_multiple_pages(sample_file):
    pages = [
        DocumentPage(
            document_id="doc-001",
            page_number=1,
            text="First page content.",
        ),
        DocumentPage(
            document_id="doc-001",
            page_number=3,
            text="Third page content.",
        ),
    ]

    service = make_service(pages)

    result = service.ingest(
        sample_file,
        document_id="doc-001",
    )

    assert result.chunk_count == 2
    assert [c.page_number for c in result.chunks] == [1, 3]


def test_ingestion_rejects_empty_documents(sample_file):
    service = make_service([])

    with pytest.raises(
        ValueError,
        match="no extractable text",
    ):
        service.ingest(
            sample_file,
            document_id="doc-001",
        )


def test_ingestion_rejects_empty_document_id(sample_file):
    service = make_service([])

    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        service.ingest(
            sample_file,
            document_id="",
        )


def test_ingestion_is_deterministic(sample_file):
    pages = [
        DocumentPage(
            document_id="doc-001",
            page_number=1,
            text="Employees receive annual leave.",
        )
    ]

    service = make_service(pages)

    first = service.ingest(sample_file, "doc-001")
    second = service.ingest(sample_file, "doc-001")

    assert first.identity == second.identity
    assert first.chunks == second.chunks


def test_ingestion_rejects_mismatched_document_id(sample_file):
    pages = [
        DocumentPage(
            document_id="another-document",
            page_number=1,
            text="Some document text.",
        )
    ]

    service = make_service(pages)

    with pytest.raises(
        ValueError,
        match="different document",
    ):
        service.ingest(
            sample_file,
            document_id="doc-001",
        )


def test_ingestion_hash_changes_when_file_changes(sample_file):
    pages = [
        DocumentPage(
            document_id="doc-001",
            page_number=1,
            text="Some document text.",
        )
    ]

    service = make_service(pages)

    first = service.ingest(sample_file, "doc-001")

    sample_file.write_bytes(b"Updated document contents")

    second = service.ingest(sample_file, "doc-001")

    assert first.content_hash != second.content_hash
    assert first.document_id == second.document_id
