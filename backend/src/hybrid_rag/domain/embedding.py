
from math import isfinite
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = Field(min_length=1)
    dimensions: int = Field(gt=0)


class EmbeddedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    vector: tuple[float, ...]

    @field_validator("vector")
    @classmethod
    def validate_vector(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        if not value:
            raise ValueError(
                "Embedding vector cannot be empty"
            )

        if not all(isfinite(number) for number in value):
            raise ValueError(
                "Embedding vector must contain finite values"
            )

        return value


class EmbeddingModel(Protocol):
    @property
    def dimensions(self) -> int:
        ...

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        ...
