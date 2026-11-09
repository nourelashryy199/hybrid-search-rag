
from typing import Protocol

from hybrid_rag.domain.generation import GeneratedAnswer
from hybrid_rag.domain.reranking import RerankedResult


class RetrievalPipelinePort(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RerankedResult]:
        ...


class GroundedAnswerPort(Protocol):
    def answer(
        self,
        question: str,
        results: list[RerankedResult],
    ) -> GeneratedAnswer:
        ...


class RAGService:
    """Orchestrate retrieval and grounded answer generation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipelinePort,
        answer_service: GroundedAnswerPort,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.answer_service = answer_service

    def ask(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        results = self.retrieval_pipeline.search(
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        return self.answer_service.answer(
            question=question,
            results=results,
        )
