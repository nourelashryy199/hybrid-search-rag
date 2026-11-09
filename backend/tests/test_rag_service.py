
import pytest

from hybrid_rag.application.rag_service import RAGService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.generation import (
    AnswerCitation,
    GeneratedAnswer,
)
from hybrid_rag.domain.reranking import RerankedResult


def make_result(chunk_id="leave-001"):
    return RerankedResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="handbook",
            text="Employees receive 25 days of annual leave.",
            chunk_index=0,
            page_number=4,
        ),
        score=0.95,
        original_rank=1,
    )


def make_answer():
    return GeneratedAnswer(
        answer="Employees receive 25 days of annual leave.",
        citations=(
            AnswerCitation(
                chunk_id="leave-001",
                document_id="handbook",
                page_number=4,
            ),
        ),
        abstained=False,
    )


class FakeRetrievalPipeline:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5, candidate_k=20):
        self.calls.append((query, top_k, candidate_k))
        return self.results[:top_k]


class FakeAnswerService:
    def __init__(self, answer):
        self.answer_value = answer
        self.calls = []

    def answer(self, question, results):
        self.calls.append((question, results))
        return self.answer_value


def test_rag_service_returns_generated_answer():
    retriever = FakeRetrievalPipeline([make_result()])
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    answer = service.ask(
        "How much annual leave?",
        top_k=5,
        candidate_k=20,
    )

    assert "25 days" in answer.answer
    assert answer.abstained is False
    assert len(answer.citations) == 1


def test_rag_service_passes_parameters_to_retrieval():
    retriever = FakeRetrievalPipeline([make_result()])
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    service.ask(
        "Annual leave?",
        top_k=3,
        candidate_k=12,
    )

    assert retriever.calls == [
        ("Annual leave?", 3, 12)
    ]


def test_rag_service_passes_results_to_answer_service():
    results = [make_result()]
    retriever = FakeRetrievalPipeline(results)
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    service.ask("Annual leave?")

    assert answer_service.calls == [
        ("Annual leave?", results)
    ]


def test_rag_service_supports_abstention():
    retriever = FakeRetrievalPipeline([])

    answer_service = FakeAnswerService(
        GeneratedAnswer(
            answer="Insufficient information.",
            citations=(),
            abstained=True,
        )
    )

    service = RAGService(retriever, answer_service)

    answer = service.ask("Unknown question?")

    assert answer.abstained is True
    assert answer.citations == ()


def test_rag_service_rejects_empty_question():
    retriever = FakeRetrievalPipeline([])
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    with pytest.raises(ValueError, match="Question"):
        service.ask("   ")

    assert retriever.calls == []
    assert answer_service.calls == []


def test_rag_service_rejects_invalid_top_k():
    retriever = FakeRetrievalPipeline([])
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    with pytest.raises(ValueError, match="top_k"):
        service.ask("Question?", top_k=0)


def test_rag_service_rejects_invalid_candidate_k():
    retriever = FakeRetrievalPipeline([])
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    with pytest.raises(ValueError, match="candidate_k"):
        service.ask(
            "Question?",
            top_k=5,
            candidate_k=3,
        )


def test_rag_service_propagates_retrieval_errors():
    class FailingRetriever:
        def search(self, query, top_k=5, candidate_k=20):
            raise RuntimeError("Retrieval failed")

    answer_service = FakeAnswerService(make_answer())

    service = RAGService(
        FailingRetriever(),
        answer_service,
    )

    with pytest.raises(RuntimeError, match="Retrieval failed"):
        service.ask("Question?")

    assert answer_service.calls == []


def test_rag_service_propagates_generation_errors():
    retriever = FakeRetrievalPipeline([make_result()])

    class FailingAnswerService:
        def answer(self, question, results):
            raise RuntimeError("Generation failed")

    service = RAGService(
        retriever,
        FailingAnswerService(),
    )

    with pytest.raises(RuntimeError, match="Generation failed"):
        service.ask("Question?")


def test_rag_service_preserves_citation_metadata():
    retriever = FakeRetrievalPipeline([make_result()])
    answer_service = FakeAnswerService(make_answer())

    service = RAGService(retriever, answer_service)

    answer = service.ask("Annual leave?")

    citation = answer.citations[0]

    assert citation.chunk_id == "leave-001"
    assert citation.document_id == "handbook"
    assert citation.page_number == 4
