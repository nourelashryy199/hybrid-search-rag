
from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from hybrid_rag.api.main import create_api
from hybrid_rag.application.factory import create_rag_application
from hybrid_rag.infrastructure.config import AppSettings


class FakeEmbeddingModel:
    @property
    def dimensions(self):
        return 3

    def embed_documents(self, texts):
        return [
            [
                1.0 + float("leave" in text.lower()),
                1.0 + float("security" in text.lower()),
                1.0 + float("expense" in text.lower()),
            ]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def make_pdf(text):
    document = fitz.open()

    try:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def make_settings(tmp_path):
    return AppSettings(
        qdrant_path=tmp_path / "qdrant",
        sqlite_path=tmp_path / "documents.db",
        embedding_dimensions=3,
        enable_reranking=False,
        anthropic_api_key=None,
        anthropic_model=None,
    )


def make_api(settings, monkeypatch):
    real_factory = create_rag_application

    def test_factory(settings):
        return real_factory(
            settings=settings,
            embedding_model=FakeEmbeddingModel(),
        )

    monkeypatch.setattr(
        "hybrid_rag.api.main.create_rag_application",
        test_factory,
    )

    return create_api(
        settings=settings,
        initialize_rag=True,
    )


def upload_pdf(client, content):
    return client.post(
        "/documents",
        files={
            "file": (
                "handbook.pdf",
                BytesIO(content),
                "application/pdf",
            ),
        },
    )


def test_pdf_upload_indexes_real_document(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    api = make_api(settings, monkeypatch)

    pdf = make_pdf(
        "Employees receive 25 days of annual leave."
    )

    with TestClient(api) as client:
        response = upload_pdf(client, pdf)

        assert response.status_code == 201

        data = response.json()

        assert data["action"] == "create"
        assert data["chunk_count"] >= 1
        assert len(data["document_id"]) == 64

        results = (
            client.app.state.rag_application.retrieval.search(
                "annual leave",
                top_k=5,
                candidate_k=5,
            )
        )

        assert results
        assert any(
            "25 days" in result.chunk.text
            for result in results
        )


def test_duplicate_pdf_upload_is_skipped(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    api = make_api(settings, monkeypatch)

    pdf = make_pdf(
        "Employees receive 25 days of annual leave."
    )

    with TestClient(api) as client:
        first = upload_pdf(client, pdf)
        second = upload_pdf(client, pdf)

        assert first.status_code == 201
        assert second.status_code == 201

        assert first.json()["action"] == "create"
        assert second.json()["action"] == "skip"

        assert (
            first.json()["document_id"]
            == second.json()["document_id"]
        )


def test_pdf_remains_searchable_after_restart(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)

    pdf = make_pdf(
        "Security policy requires multifactor authentication."
    )

    first_api = make_api(settings, monkeypatch)

    with TestClient(first_api) as client:
        response = upload_pdf(client, pdf)
        assert response.status_code == 201

    second_api = make_api(settings, monkeypatch)

    with TestClient(second_api) as client:
        results = (
            client.app.state.rag_application.retrieval.search(
                "security multifactor authentication",
                top_k=5,
                candidate_k=5,
            )
        )

        assert results
        assert any(
            "multifactor authentication" in result.chunk.text
            for result in results
        )


def test_pdf_upload_does_not_require_anthropic_key(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    api = make_api(settings, monkeypatch)

    pdf = make_pdf(
        "Expense claims must be submitted within 30 days."
    )

    with TestClient(api) as client:
        upload_response = upload_pdf(client, pdf)

        assert upload_response.status_code == 201

        ask_response = client.post(
            "/ask",
            json={"question": "What is the expense policy?"},
        )

        assert ask_response.status_code == 503


def test_pdf_upload_service_is_cleaned_up_on_shutdown(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    api = make_api(settings, monkeypatch)

    with TestClient(api) as client:
        assert (
            client.app.state.document_ingestion_service
            is not None
        )

    assert api.state.document_ingestion_service is None
    assert api.state.rag_application is None
