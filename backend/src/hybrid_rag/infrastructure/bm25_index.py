import re

from rank_bm25 import BM25Okapi

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._model: BM25Okapi | None = None

    def add(self, chunks: list[DocumentChunk]) -> None:
        updated = {
            chunk.chunk_id: chunk
            for chunk in self._chunks
        }

        for chunk in chunks:
            updated[chunk.chunk_id] = chunk

        self._chunks = list(updated.values())

        if not self._chunks:
            self._model = None
            return

        tokenized = [
            tokenize(chunk.text)
            for chunk in self._chunks
        ]

        self._model = BM25Okapi(tokenized)


    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(query)

        if not query_tokens or self._model is None:
            return []

        scores = self._model.get_scores(query_tokens)
        query_terms = set(query_tokens)

        results = [
            SearchResult(
                chunk=chunk,
                score=float(score),
            )
            for chunk, score in zip(self._chunks, scores)
            if query_terms.intersection(tokenize(chunk.text))
        ]

        return sorted(
            results,
            key=lambda result: (
                -result.score,
                result.chunk.chunk_id,
            ),
        )[:top_k]
