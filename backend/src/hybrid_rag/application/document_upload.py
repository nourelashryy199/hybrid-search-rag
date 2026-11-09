
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from hybrid_rag.domain.indexing import IndexingAction


class DocumentIngestor(Protocol):
    def ingest(
        self,
        file_path: str | Path,
        document_id: str,
    ):
        ...


class DocumentIndexer(Protocol):
    def index(
        self,
        identity,
        chunks,
    ) -> IndexingAction:
        ...


class DocumentUploadService:
    """Coordinate PDF ingestion and persistent indexing."""

    def __init__(
        self,
        ingestor: DocumentIngestor,
        indexer: DocumentIndexer,
    ) -> None:
        self.ingestor = ingestor
        self.indexer = indexer

    def ingest(
        self,
        file_path: Path,
        filename: str,
    ) -> dict:
        # Generate a stable ID from the PDF's content.
        hasher = sha256()

        with file_path.open("rb") as source:
            for block in iter(
                lambda: source.read(1024 * 1024),
                b"",
            ):
                hasher.update(block)

        document_id = hasher.hexdigest()

        result = self.ingestor.ingest(
            file_path=file_path,
            document_id=document_id,
        )

        if not result.chunks:
            raise ValueError(
                "The PDF contains no indexable text"
            )

        action = self.indexer.index(
            identity=result.identity,
            chunks=list(result.chunks),
        )

        return {
            "document_id": result.identity.document_id,
            "filename": filename,
            "chunk_count": len(result.chunks),
            "action": action,
        }
