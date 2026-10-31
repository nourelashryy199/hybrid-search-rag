
from collections import defaultdict

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.retrieval import SearchResult


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF constant k must be positive")

        self.k = k

    def fuse(
        self,
        rankings: list[list[SearchResult]],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}

        for ranking in rankings:
            seen: set[str] = set()

            for rank, result in enumerate(ranking, start=1):
                chunk_id = result.chunk.chunk_id

                if chunk_id in seen:
                    continue

                seen.add(chunk_id)

                existing = chunks.get(chunk_id)

                if existing is not None and existing != result.chunk:
                    raise ValueError(
                        "Conflicting chunks share the same chunk ID"
                    )

                chunks[chunk_id] = result.chunk
                scores[chunk_id] += 1.0 / (self.k + rank)

        ordered_ids = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                chunk_id,
            ),
        )

        return [
            SearchResult(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
            )
            for chunk_id in ordered_ids[:top_k]
        ]
