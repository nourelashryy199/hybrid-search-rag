
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import (
    EmbeddedChunk,
    EmbeddingModel,
)


class EmbeddingService:
    def __init__(
        self,
        model: EmbeddingModel,
    ) -> None:
        self.model = model

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        vectors = self.model.embed_documents(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding model returned an incorrect "
                "number of vectors"
            )

        embedded_chunks = []

        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.model.dimensions:
                raise ValueError(
                    f"Expected {self.model.dimensions} "
                    f"dimensions, received {len(vector)}"
                )

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    vector=tuple(vector),
                )
            )

        return embedded_chunks
