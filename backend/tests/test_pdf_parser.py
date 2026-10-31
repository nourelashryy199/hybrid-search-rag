
import pymupdf
import pytest

from hybrid_rag.infrastructure.pdf_parser import PDFParser


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text(
            (72, 72),
            "Employees receive 25 days of annual leave."
        )

        pdf.new_page()

        third = pdf.new_page()
        third.insert_text(
            (72, 72),
            "Remote work requires manager approval."
        )

        pdf.save(path)

    return path


def test_pdf_parser_extracts_text(sample_pdf):
    parser = PDFParser()

    pages = parser.parse(
        sample_pdf,
        document_id="policy-001",
    )

    assert len(pages) == 2

    assert "25 days" in pages[0].text
    assert "manager approval" in pages[1].text


def test_pdf_parser_preserves_page_numbers(sample_pdf):
    parser = PDFParser()

    pages = parser.parse(
        sample_pdf,
        document_id="policy-001",
    )

    assert pages[0].page_number == 1
    assert pages[1].page_number == 3


def test_pdf_parser_rejects_missing_file(tmp_path):
    parser = PDFParser()

    with pytest.raises(FileNotFoundError):
        parser.parse(
            tmp_path / "missing.pdf",
            document_id="policy-001",
        )


def test_pdf_parser_rejects_non_pdf(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Not a PDF")

    parser = PDFParser()

    with pytest.raises(ValueError):
        parser.parse(
            path,
            document_id="policy-001",
        )


def test_pdf_parser_rejects_empty_document_id(sample_pdf):
    parser = PDFParser()

    with pytest.raises(ValueError):
        parser.parse(
            sample_pdf,
            document_id="",
        )
