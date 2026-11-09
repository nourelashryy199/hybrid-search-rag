
from typing import Protocol

from hybrid_rag.domain.generation import (
    EvidencePassage,
    GeneratedAnswer,
)
from hybrid_rag.domain.reranking import RerankedResult


class EvidenceAnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        evidence: list[EvidencePassage],
    ) -> GeneratedAnswer:
        ...


class GroundedAnswerService:
    """Prepare retrieved evidence and validate answer references."""

    def __init__(
        self,
        generator: EvidenceAnswerGenerator,
    ) -> None:
        self.generator = generator

    def answer(
        self,
        question: str,
        results: list[RerankedResult],
    ) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question must not be empty")

        if not results:
            return GeneratedAnswer(
                answer=(
                    "The provided documents do not contain "
                    "enough information to answer this question."
                ),
                citations=(),
                abstained=True,
            )

        evidence: list[EvidencePassage] = []
        seen_chunk_ids: set[str] = set()

        for result in results:
            chunk = result.chunk

            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate retrieved chunk ID: {chunk.chunk_id}"
                )

            seen_chunk_ids.add(chunk.chunk_id)

            evidence.append(
                EvidencePassage(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                )
            )

        generated = self.generator.generate(
            question=question,
            evidence=evidence,
        )

        if generated.abstained:
            if generated.citations:
                raise ValueError(
                    "Abstained answers must not contain citations"
                )
            return generated

        if not generated.citations:
            raise ValueError(
                "Non-abstained answers must include citations"
            )

        evidence_lookup = {
            passage.chunk_id: passage
            for passage in evidence
        }

        cited_ids: set[str] = set()

        for citation in generated.citations:
            if citation.chunk_id in cited_ids:
                raise ValueError(
                    "Duplicate answer citation"
                )

            cited_ids.add(citation.chunk_id)

            passage = evidence_lookup.get(citation.chunk_id)

            if passage is None:
                raise ValueError(
                    f"Citation references unknown chunk: "
                    f"{citation.chunk_id}"
                )

            if (
                citation.document_id != passage.document_id
                or citation.page_number != passage.page_number
            ):
                raise ValueError(
                    "Citation metadata does not match retrieved evidence"
                )

        return generated
