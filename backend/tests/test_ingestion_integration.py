
import pymupdf

from hybrid_rag.application.chunking import (
    ChunkingConfig,
    TextChunker,
)
from hybrid_rag.application.ingestion import (
    DocumentIngestionService,
)
from hybrid_rag.infrastructure.pdf_parser import PDFParser


def test_pdf_ingestion_end_to_end(tmp_path):
    pdf_path = tmp_path / "employee_policy.pdf"

    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text(
            (72, 72),
            "Employees receive 25 days of annual leave.",
        )

        second_page = pdf.new_page()
        second_page.insert_text(
            (72, 72),
            "Remote work requires manager approval.",
        )

        pdf.save(pdf_path)

    service = DocumentIngestionService(
        parser=PDFParser(),
        chunker=TextChunker(
            ChunkingConfig(
                chunk_size=100,
                chunk_overlap=20,
            )
        ),
    )

    result = service.ingest(
        file_path=pdf_path,
        document_id="employee-policy-001",
    )

    assert result.document_id == "employee-policy-001"
    assert result.filename == "employee_policy.pdf"
    assert result.chunk_count == 2

    assert result.chunks[0].page_number == 1
    assert result.chunks[1].page_number == 2

    assert "25 days" in result.chunks[0].text
    assert "manager approval" in result.chunks[1].text

    assert result.chunks[0].chunk_id != result.chunks[1].chunk_id

    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]
