
import pytest
from pydantic import ValidationError

from hybrid_rag.infrastructure.config import AppSettings


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    for key in (
        "HYBRID_RAG_QDRANT_PATH",
        "HYBRID_RAG_SQLITE_PATH",
        "HYBRID_RAG_ENABLE_RERANKING",
        "HYBRID_RAG_COHERE_API_KEY",
        "HYBRID_RAG_ANTHROPIC_API_KEY",
        "HYBRID_RAG_ANTHROPIC_MODEL",
        "HYBRID_RAG_RETRIEVAL_TOP_K",
        "HYBRID_RAG_RETRIEVAL_CANDIDATE_K",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_configuration(clean_environment):
    settings = AppSettings()

    assert settings.embedding_dimensions == 384
    assert settings.enable_reranking is False
    assert settings.retrieval_top_k == 5
    assert settings.retrieval_candidate_k == 20


def test_environment_overrides_configuration(
    clean_environment,
    monkeypatch,
):
    monkeypatch.setenv(
        "HYBRID_RAG_RETRIEVAL_TOP_K",
        "3",
    )
    monkeypatch.setenv(
        "HYBRID_RAG_RETRIEVAL_CANDIDATE_K",
        "12",
    )

    settings = AppSettings()

    assert settings.retrieval_top_k == 3
    assert settings.retrieval_candidate_k == 12


def test_reranking_requires_cohere_key(clean_environment):
    with pytest.raises(ValidationError, match="Cohere API key"):
        AppSettings(enable_reranking=True)


def test_reranking_accepts_cohere_key(clean_environment):
    settings = AppSettings(
        enable_reranking=True,
        cohere_api_key="test-key",
    )

    assert settings.enable_reranking is True


def test_invalid_embedding_dimensions(clean_environment):
    with pytest.raises(ValidationError, match="dimensions"):
        AppSettings(embedding_dimensions=0)


def test_invalid_retrieval_top_k(clean_environment):
    with pytest.raises(ValidationError, match="top_k"):
        AppSettings(retrieval_top_k=0)


def test_candidate_k_must_not_be_smaller(clean_environment):
    with pytest.raises(ValidationError, match="candidate_k"):
        AppSettings(
            retrieval_top_k=10,
            retrieval_candidate_k=5,
        )


def test_blank_optional_credentials_become_none(
    clean_environment,
):
    settings = AppSettings(
        cohere_api_key="  ",
        anthropic_api_key="  ",
        anthropic_model="  ",
    )

    assert settings.cohere_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.anthropic_model is None


def test_anthropic_key_is_optional_at_configuration_time(
    clean_environment,
):
    settings = AppSettings()

    assert settings.anthropic_api_key is None


def test_invalid_collection_name(clean_environment):
    with pytest.raises(ValidationError, match="name"):
        AppSettings(qdrant_collection="   ")
