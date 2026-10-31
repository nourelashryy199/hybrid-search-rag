
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hybrid_rag.application.chunking import TextChunker
from hybrid_rag.domain.document import DocumentChunk, DocumentPage
from hybrid_rag.domain.identity import (
    DocumentIdentity,
    create_document_identity,
)


class DocumentParser(Protocol):
    def parse(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> list[DocumentPage]:
        ...


@dataclass(frozen=True)
class IngestionResult:
    identity: DocumentIdentity
    filename: str
    chunks: tuple[DocumentChunk, ...]

    @property
    def document_id(self) -> str:
        return self.identity.document_id

    @property
    def content_hash(self) -> str:
        return self.identity.content_hash

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

        identity = create_document_identity(
            file_path=path,
            document_id=document_id,
        )

        pages = self.parser.parse(
            file_path=path,
            document_id=document_id,
        )

        if any(
            page.document_id != document_id
            for page in pages
        ):
            raise ValueError(
                "Parser returned pages belonging "
                "to a different document"
            )

        chunks = self.chunker.chunk(pages)

        if not chunks:
            raise ValueError(
                "Document contains no extractable text"
            )

        return IngestionResult(
            identity=identity,
            filename=path.name,
            chunks=tuple(chunks),
        )
