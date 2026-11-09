
from fastapi.testclient import TestClient

from hybrid_rag.api.main import create_api


class FakeDocumentUploadService:
    def __init__(self):
        self.calls = []

    def ingest(self, file_path, filename):
        self.calls.append(
            (
                filename,
                file_path.read_bytes(),
            )
        )

        return {
            "document_id": "test-document",
            "filename": filename,
            "chunk_count": 2,
            "action": "create",
        }


def make_client(service=None):
    api = create_api()

    api.state.document_ingestion_service = service

    return TestClient(api)


def test_pdf_upload_succeeds():
    service = FakeDocumentUploadService()

    with make_client(service) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "handbook.pdf",
                    b"%PDF-1.4\nsample",
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 201
    assert response.json()["document_id"] == "test-document"
    assert response.json()["chunk_count"] == 2
    assert service.calls[0][0] == "handbook.pdf"


def test_pdf_upload_rejects_non_pdf_filename():
    service = FakeDocumentUploadService()

    with make_client(service) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "notes.txt",
                    b"plain text",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 415
    assert service.calls == []


def test_pdf_upload_rejects_invalid_signature():
    service = FakeDocumentUploadService()

    with make_client(service) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "fake.pdf",
                    b"not a PDF",
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 415
    assert service.calls == []


def test_pdf_upload_requires_ingestion_service():
    with make_client() as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "handbook.pdf",
                    b"%PDF-1.4\nsample",
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 503


def test_pdf_upload_rejects_missing_file():
    service = FakeDocumentUploadService()

    with make_client(service) as client:
        response = client.post(
            "/documents",
        )

    assert response.status_code == 422
    assert service.calls == []


def test_pdf_upload_rejects_oversized_file():
    service = FakeDocumentUploadService()

    with make_client(service) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "large.pdf",
                    b"%PDF-" + b"x" * (20 * 1024 * 1024),
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 413
    assert service.calls == []
