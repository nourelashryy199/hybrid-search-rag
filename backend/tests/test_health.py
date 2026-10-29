from fastapi.testclient import TestClient
from hybrid_rag.api.main import app

def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"healthy"}

