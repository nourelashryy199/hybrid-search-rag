
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


def test_ingestion_returns_chunks():
    pages = [
        DocumentPage(
            document_id="doc-001",
            page_number=1,
            text="Employees receive annual leave.",
        )
    ]

    service = make_service(pages)

    result = service.ingest(
        "policy.pdf",
        document_id="doc-001",
    )

    assert result.document_id == "doc-001"
    assert result.filename == "policy.pdf"
    assert result.chunk_count == 1
    assert result.chunks[0].page_number == 1


def test_ingestion_preserves_multiple_pages():
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
        "policy.pdf",
        document_id="doc-001",
    )

    assert result.chunk_count == 2
    assert [c.page_number for c in result.chunks] == [1, 3]


def test_ingestion_rejects_empty_documents():
    service = make_service([])

    with pytest.raises(
        ValueError,
        match="no extractable text",
    ):
        service.ingest(
            "empty.pdf",
            document_id="doc-001",
        )


def test_ingestion_rejects_empty_document_id():
    service = make_service([])

    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        service.ingest(
            "policy.pdf",
            document_id="",
        )


def test_ingestion_is_deterministic():
    pages = [
        DocumentPage(
            document_id="doc-001",
            page_number=1,
            text="Employees receive annual leave.",
        )
    ]

    service = make_service(pages)

    first = service.ingest("policy.pdf", "doc-001")
    second = service.ingest("policy.pdf", "doc-001")

    assert first.chunks == second.chunks
