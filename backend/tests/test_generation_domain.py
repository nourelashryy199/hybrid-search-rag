
import pytest
from pydantic import ValidationError

from hybrid_rag.domain.generation import (
    AnswerCitation,
    EvidencePassage,
    GeneratedAnswer,
)


def make_evidence() -> EvidencePassage:
    return EvidencePassage(
        chunk_id="chunk-001",
        document_id="handbook",
        text="Employees receive 25 days of annual leave.",
        page_number=4,
    )


def make_citation() -> AnswerCitation:
    return AnswerCitation(
        chunk_id="chunk-001",
        document_id="handbook",
        page_number=4,
    )


def test_evidence_passage_creation():
    evidence = make_evidence()

    assert evidence.chunk_id == "chunk-001"
    assert evidence.document_id == "handbook"
    assert evidence.page_number == 4


def test_evidence_rejects_empty_text():
    with pytest.raises(ValidationError):
        EvidencePassage(
            chunk_id="chunk-001",
            document_id="handbook",
            text="   ",
            page_number=4,
        )


def test_evidence_rejects_invalid_page_number():
    with pytest.raises(ValidationError):
        EvidencePassage(
            chunk_id="chunk-001",
            document_id="handbook",
            text="Policy content",
            page_number=0,
        )


def test_evidence_is_immutable():
    evidence = make_evidence()

    with pytest.raises(ValidationError):
        evidence.text = "Modified content"


def test_citation_creation():
    citation = make_citation()

    assert citation.chunk_id == "chunk-001"
    assert citation.page_number == 4


def test_citation_rejects_empty_chunk_id():
    with pytest.raises(ValidationError):
        AnswerCitation(
            chunk_id="",
            document_id="handbook",
            page_number=4,
        )


def test_generated_answer_with_citation():
    answer = GeneratedAnswer(
        answer="Employees receive 25 days of annual leave.",
        citations=(make_citation(),),
        abstained=False,
    )

    assert answer.abstained is False
    assert len(answer.citations) == 1


def test_generated_answer_can_abstain():
    answer = GeneratedAnswer(
        answer="The provided documents do not contain this information.",
        citations=(),
        abstained=True,
    )

    assert answer.abstained is True
    assert answer.citations == ()


def test_abstained_answer_rejects_citations():
    with pytest.raises(
        ValidationError,
        match="Abstained answers",
    ):
        GeneratedAnswer(
            answer="Insufficient evidence.",
            citations=(make_citation(),),
            abstained=True,
        )


def test_generated_answer_rejects_empty_text():
    with pytest.raises(ValidationError):
        GeneratedAnswer(
            answer="   ",
            citations=(),
            abstained=False,
        )


def test_generated_answer_defaults_to_no_citations():
    answer = GeneratedAnswer(
        answer="A response awaiting evidence validation."
    )

    assert answer.citations == ()
    assert answer.abstained is False


def test_generated_answer_is_immutable():
    answer = GeneratedAnswer(
        answer="Employees receive 25 days of leave."
    )

    with pytest.raises(ValidationError):
        answer.answer = "Modified answer"
