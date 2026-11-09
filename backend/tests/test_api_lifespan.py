
from fastapi.testclient import TestClient

from hybrid_rag.api.main import create_api
from hybrid_rag.infrastructure.config import AppSettings


def test_api_initializes_persistent_application(
    tmp_path,
    monkeypatch,
):
    class FakeApplication:
        def __init__(self):
            self.rag = None
            self.indexing = object()
            self.closed = False

        def close(self):
            self.closed = True

    fake_application = FakeApplication()

    def fake_factory(settings):
        return fake_application

    monkeypatch.setattr(
        "hybrid_rag.api.main.create_rag_application",
        fake_factory,
    )

    settings = AppSettings(
        qdrant_path=tmp_path / "qdrant",
        sqlite_path=tmp_path / "documents.db",
    )

    api = create_api(
        settings=settings,
        initialize_rag=True,
    )

    with TestClient(api) as client:
        assert client.app.state.rag_application is fake_application
        assert client.app.state.indexing_service is fake_application.indexing

        response = client.get("/health")
        assert response.status_code == 200

    assert fake_application.closed is True


def test_api_returns_503_without_generator(
    tmp_path,
    monkeypatch,
):
    class FakeApplication:
        rag = None
        indexing = object()

        def close(self):
            pass

    monkeypatch.setattr(
        "hybrid_rag.api.main.create_rag_application",
        lambda settings: FakeApplication(),
    )

    api = create_api(
        settings=AppSettings(
            qdrant_path=tmp_path / "qdrant",
            sqlite_path=tmp_path / "documents.db",
        ),
        initialize_rag=True,
    )

    with TestClient(api) as client:
        response = client.post(
            "/ask",
            json={"question": "What is the leave policy?"},
        )

    assert response.status_code == 503


def test_api_closes_application_after_request_error(
    tmp_path,
    monkeypatch,
):
    class FakeApplication:
        rag = None
        indexing = object()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    application = FakeApplication()

    monkeypatch.setattr(
        "hybrid_rag.api.main.create_rag_application",
        lambda settings: application,
    )

    api = create_api(
        settings=AppSettings(
            qdrant_path=tmp_path / "qdrant",
            sqlite_path=tmp_path / "documents.db",
        ),
        initialize_rag=True,
    )

    with TestClient(api) as client:
        response = client.post(
            "/ask",
            json={"question": "What is the leave policy?"},
        )
        assert response.status_code == 503

    assert application.closed is True


def test_api_can_start_without_persistent_initialization():
    api = create_api()

    with TestClient(api) as client:
        response = client.get("/health")

    assert response.status_code == 200
