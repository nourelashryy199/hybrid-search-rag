
import pytest

from hybrid_rag.application.grounded_answer import (
    GroundedAnswerService,
)
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.generation import (
    AnswerCitation,
    GeneratedAnswer,
)
from hybrid_rag.domain.reranking import RerankedResult


def make_result(
    chunk_id: str,
    document_id: str = "handbook",
    page_number: int = 4,
) -> RerankedResult:
    return RerankedResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=f"Evidence for {chunk_id}",
            chunk_index=0,
            page_number=page_number,
        ),
        score=0.9,
        original_rank=1,
    )


def make_citation(
    chunk_id: str = "leave-001",
    document_id: str = "handbook",
    page_number: int = 4,
) -> AnswerCitation:
    return AnswerCitation(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
    )


class FakeGenerator:
    def __init__(self, answer: GeneratedAnswer):
        self.answer_value = answer
        self.calls = []

    def generate(self, question, evidence):
        self.calls.append((question, evidence))
        return self.answer_value


def test_valid_answer_passes():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Employees receive 25 days of leave.",
            citations=(make_citation(),),
        )
    )

    service = GroundedAnswerService(generator)

    answer = service.answer(
        "How much leave?",
        [make_result("leave-001")],
    )

    assert answer.abstained is False
    assert answer.citations[0].chunk_id == "leave-001"


def test_retrieved_chunks_are_converted_to_evidence():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="The policy is described in the handbook.",
            citations=(make_citation(),),
        )
    )

    service = GroundedAnswerService(generator)

    service.answer(
        "What is the policy?",
        [make_result("leave-001")],
    )

    question, evidence = generator.calls[0]

    assert question == "What is the policy?"
    assert len(evidence) == 1
    assert evidence[0].chunk_id == "leave-001"
    assert evidence[0].document_id == "handbook"
    assert evidence[0].page_number == 4


def test_no_evidence_causes_abstention():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="This should never be returned.",
        )
    )

    service = GroundedAnswerService(generator)

    answer = service.answer("Unknown question?", [])

    assert answer.abstained is True
    assert answer.citations == ()
    assert generator.calls == []


def test_empty_question_is_rejected():
    generator = FakeGenerator(
        GeneratedAnswer(answer="Unused")
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="Question"):
        service.answer("   ", [make_result("leave-001")])


def test_unknown_citation_is_rejected():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="An answer.",
            citations=(make_citation("invented"),),
        )
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="unknown chunk"):
        service.answer(
            "Question?",
            [make_result("leave-001")],
        )


def test_wrong_document_id_is_rejected():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="An answer.",
            citations=(
                make_citation(document_id="wrong-document"),
            ),
        )
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="metadata"):
        service.answer(
            "Question?",
            [make_result("leave-001")],
        )


def test_wrong_page_number_is_rejected():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="An answer.",
            citations=(
                make_citation(page_number=99),
            ),
        )
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="metadata"):
        service.answer(
            "Question?",
            [make_result("leave-001")],
        )


def test_uncited_answer_is_rejected():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Unsupported answer.",
            citations=(),
            abstained=False,
        )
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="include citations"):
        service.answer(
            "Question?",
            [make_result("leave-001")],
        )


def test_duplicate_retrieved_chunks_are_rejected():
    generator = FakeGenerator(
        GeneratedAnswer(answer="Unused")
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="Duplicate retrieved"):
        service.answer(
            "Question?",
            [
                make_result("leave-001"),
                make_result("leave-001"),
            ],
        )


def test_duplicate_citations_are_rejected():
    citation = make_citation()

    generator = FakeGenerator(
        GeneratedAnswer(
            answer="An answer.",
            citations=(citation, citation),
        )
    )

    service = GroundedAnswerService(generator)

    with pytest.raises(ValueError, match="Duplicate answer citation"):
        service.answer(
            "Question?",
            [make_result("leave-001")],
        )


def test_valid_abstention_passes():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Insufficient evidence.",
            citations=(),
            abstained=True,
        )
    )

    service = GroundedAnswerService(generator)

    answer = service.answer(
        "Question?",
        [make_result("leave-001")],
    )

    assert answer.abstained is True
    assert answer.citations == ()


def test_multiple_valid_citations_pass():
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Both documents describe the policy.",
            citations=(
                make_citation("leave-001"),
                make_citation(
                    "leave-002",
                    document_id="policy",
                    page_number=2,
                ),
            ),
        )
    )

    service = GroundedAnswerService(generator)

    answer = service.answer(
        "Compare the policies.",
        [
            make_result("leave-001"),
            make_result(
                "leave-002",
                document_id="policy",
                page_number=2,
            ),
        ],
    )

    assert len(answer.citations) == 2
