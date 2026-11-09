
from fastapi.testclient import TestClient

from hybrid_rag.api.main import create_api
from hybrid_rag.domain.generation import (
    AnswerCitation,
    GeneratedAnswer,
)


class FakeRAGService:
    def __init__(self):
        self.calls = []

    def ask(self, question, top_k=5, candidate_k=20):
        self.calls.append((question, top_k, candidate_k))

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


def test_ask_returns_answer_and_citations():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={"question": "How much annual leave?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert "25 days" in data["answer"]
    assert data["abstained"] is False
    assert data["citations"][0]["chunk_id"] == "leave-001"


def test_ask_forwards_retrieval_parameters():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Annual leave?",
                "top_k": 3,
                "candidate_k": 12,
            },
        )

    assert response.status_code == 200
    assert service.calls == [
        ("Annual leave?", 3, 12)
    ]


def test_ask_rejects_empty_question():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={"question": ""},
        )

    assert response.status_code == 422
    assert service.calls == []


def test_ask_rejects_invalid_top_k():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Annual leave?",
                "top_k": 0,
            },
        )

    assert response.status_code == 422
    assert service.calls == []


def test_ask_rejects_candidate_k_smaller_than_top_k():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Annual leave?",
                "top_k": 10,
                "candidate_k": 5,
            },
        )

    assert response.status_code == 422
    assert service.calls == []


def test_ask_returns_503_without_rag_service():
    with TestClient(create_api()) as client:
        response = client.post(
            "/ask",
            json={"question": "Annual leave?"},
        )

    assert response.status_code == 503


def test_health_endpoint_still_works():
    with TestClient(create_api()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_rejects_unknown_request_fields():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Annual leave?",
                "unexpected": "value",
            },
        )

    assert response.status_code == 422
    assert service.calls == []


def test_ask_rejects_whitespace_question():
    service = FakeRAGService()

    with TestClient(create_api(service)) as client:
        response = client.post(
            "/ask",
            json={"question": "   "},
        )

    assert response.status_code == 422
    assert service.calls == []


def test_ask_supports_abstention():
    class AbstainingService:
        def ask(self, question, top_k=5, candidate_k=20):
            return GeneratedAnswer(
                answer="Insufficient evidence.",
                citations=(),
                abstained=True,
            )

    with TestClient(create_api(AbstainingService())) as client:
        response = client.post(
            "/ask",
            json={"question": "Unknown policy?"},
        )

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []
