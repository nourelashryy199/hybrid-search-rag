from math import sqrt

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.retrieval import SearchResult


def cosine_similarity(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    if not first or len(first) != len(second):
        raise ValueError("Vector dimensions must match")

    dot_product = sum(a * b for a, b in zip(first, second))

    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))

    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cannot compare zero vectors")

    return dot_product / (first_norm * second_norm)


class InMemoryVectorIndex:
    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        self.dimensions = dimensions
        self._entries: dict[
            str, tuple[DocumentChunk, tuple[float, ...]]
        ] = {}

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[EmbeddedChunk],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must have equal lengths"
            )

        pending = {}

        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_id != embedding.chunk_id:
                raise ValueError("Chunk and embedding IDs do not match")

            if len(embedding.vector) != self.dimensions:
                raise ValueError("Embedding dimensions do not match index")

            if not any(embedding.vector):
                raise ValueError("Cannot index zero vectors")

            if chunk.chunk_id in pending:
                raise ValueError("Duplicate chunk ID in batch")

            pending[chunk.chunk_id] = (chunk, embedding.vector)

        self._entries.update(pending)

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._entries.pop(chunk_id, None)


    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if len(query_vector) != self.dimensions:
            raise ValueError("Query dimensions do not match index")

        if not all(
            isinstance(value, (int, float))
            for value in query_vector
        ):
            raise ValueError("Query vector must be numeric")

        query = tuple(float(value) for value in query_vector)

        if not all(__import__("math").isfinite(value) for value in query):
            raise ValueError("Query vector must contain finite values")

        if not any(query):
            raise ValueError("Query vector cannot be zero")

        results = [
            SearchResult(
                chunk=chunk,
                score=cosine_similarity(query, vector),
            )
            for chunk, vector in self._entries.values()
        ]

        return sorted(
            results,
            key=lambda result: (
                -result.score,
                result.chunk.chunk_id,
            ),
        )[:top_k]
