
from pathlib import Path

import pymupdf

from hybrid_rag.domain.document import DocumentPage


class PDFParser:
    def parse(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> list[DocumentPage]:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Unsupported file type. Expected PDF."
            )

        if not document_id:
            raise ValueError(
                "document_id cannot be empty"
            )

        pages = []

        with pymupdf.open(path) as pdf:
            if not pdf.is_pdf:
                raise ValueError("File is not a valid PDF")

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if not text:
                    continue

                pages.append(
                    DocumentPage(
                        document_id=document_id,
                        page_number=page_index + 1,
                        text=text,
                    )
                )

        return pages
