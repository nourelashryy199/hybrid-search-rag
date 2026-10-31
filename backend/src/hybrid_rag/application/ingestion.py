
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hybrid_rag.application.chunking import TextChunker
from hybrid_rag.domain.document import DocumentChunk, DocumentPage


class DocumentParser(Protocol):
    def parse(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> list[DocumentPage]:
        ...


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    filename: str
    chunks: tuple[DocumentChunk, ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class DocumentIngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: TextChunker,
    ) -> None:
        self.parser = parser
        self.chunker = chunker

    def ingest(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> IngestionResult:
        path = Path(file_path)

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        pages = self.parser.parse(
            file_path=path,
            document_id=document_id,
        )

        chunks = self.chunker.chunk(pages)

        if not chunks:
            raise ValueError(
                "Document contains no extractable text"
            )

        return IngestionResult(
            document_id=document_id,
            filename=path.name,
            chunks=tuple(chunks),
        )
