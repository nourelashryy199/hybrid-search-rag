
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-based configuration for the RAG application."""

    model_config = SettingsConfigDict(
        env_prefix="HYBRID_RAG_",
        env_file=".env",
        extra="ignore",
    )

    qdrant_path: Path = Path("data/qdrant_storage")
    sqlite_path: Path = Path("data/documents.db")

    qdrant_collection: str = "hybrid_rag_chunks"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dimensions: int = 384

    enable_reranking: bool = False
    cohere_api_key: str | None = None
    cohere_model: str = "rerank-v3.5"

    anthropic_api_key: str | None = None
    anthropic_model: str | None = None

    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20

    @field_validator(
        "qdrant_collection",
        "embedding_model",
        "cohere_model",
    )
    @classmethod
    def validate_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Configuration name must not be empty")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Embedding dimensions must be positive")
        return value

    @field_validator(
        "cohere_api_key",
        "anthropic_api_key",
        "anthropic_model",
    )
    @classmethod
    def normalize_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")

        if self.retrieval_candidate_k < self.retrieval_top_k:
            raise ValueError(
                "retrieval_candidate_k must be >= retrieval_top_k"
            )

        if self.enable_reranking and not self.cohere_api_key:
            raise ValueError(
                "Cohere API key required when reranking is enabled"
            )

        return self
