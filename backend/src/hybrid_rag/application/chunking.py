
from dataclasses import dataclass

from hybrid_rag.domain.document import (
    DocumentChunk,
    DocumentPage,
    generate_chunk_id,
)


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative "
                "and smaller than chunk_size"
            )


class TextChunker:
    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()

    def chunk(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentChunk]:
        chunks = []
        chunk_index = 0
        step = (
            self.config.chunk_size
            - self.config.chunk_overlap
        )

        for page in pages:
            text = page.text.strip()

            if not text:
                continue

            for start in range(0, len(text), step):
                chunk_text = text[
                    start:start + self.config.chunk_size
                ].strip()

                if not chunk_text:
                    continue

                chunks.append(
                    DocumentChunk(
                        chunk_id=generate_chunk_id(
                            page.document_id,
                            chunk_index,
                            chunk_text,
                        ),
                        document_id=page.document_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                    )
                )

                chunk_index += 1

                if start + self.config.chunk_size >= len(text):
                    break

        return chunks
