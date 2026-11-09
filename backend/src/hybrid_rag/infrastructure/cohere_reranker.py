
import math
from typing import Any

from hybrid_rag.domain.document import DocumentChunk


class CohereReranker:
    """Adapt Cohere Rerank to our application reranker contract."""

    def __init__(
        self,
        client: Any,
        model: str = "rerank-v3.5",
    ) -> None:
        if client is None:
            raise ValueError("Cohere client is required")

        if not model.strip():
            raise ValueError("Model name must not be empty")

        self.client = client
        self.model = model

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int,
    ) -> list[float]:
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not chunks:
            return []

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[chunk.text for chunk in chunks],
            top_n=len(chunks),
        )

        scores: list[float | None] = [None] * len(chunks)

        for result in response.results:
            index = result.index
            score = result.relevance_score

            if type(index) is not int:
                raise ValueError("Invalid Cohere result index")

            if index < 0 or index >= len(chunks):
                raise ValueError(
                    f"Cohere returned an invalid index: {index}"
                )

            if scores[index] is not None:
                raise ValueError(
                    f"Duplicate Cohere result index: {index}"
                )

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(
                    "Cohere returned an invalid relevance score"
                )

            scores[index] = float(score)

        if any(score is None for score in scores):
            raise ValueError(
                "Cohere did not return a score for every chunk"
            )

        return [float(score) for score in scores]
