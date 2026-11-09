
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.domain.indexing import IndexingAction


MAX_PDF_SIZE = 20 * 1024 * 1024

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    chunk_count: int = Field(ge=1)
    action: IndexingAction


class PDFIngestionPort(Protocol):
    def ingest(
        self,
        file_path: Path,
        filename: str,
    ) -> DocumentUploadResponse:
        ...


def get_ingestion_service(request: Request) -> PDFIngestionPort:
    service = getattr(
        request.app.state,
        "document_ingestion_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Document ingestion is not configured",
        )

    return service


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=201,
)
def upload_document(
    file: UploadFile = File(...),
    service: PDFIngestionPort = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported",
        )

    try:
        with TemporaryDirectory() as temporary_directory:
            file_path = (
                Path(temporary_directory) / "uploaded.pdf"
            )

            total_size = 0

            with file_path.open("wb") as destination:
                while True:
                    chunk = file.file.read(1024 * 1024)

                    if not chunk:
                        break

                    total_size += len(chunk)

                    if total_size > MAX_PDF_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF exceeds the 20 MB limit",
                        )

                    destination.write(chunk)

            with file_path.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise HTTPException(
                        status_code=415,
                        detail="Uploaded file is not a valid PDF",
                    )

            return service.ingest(
                file_path=file_path,
                filename=filename,
            )

    finally:
        file.file.close()
