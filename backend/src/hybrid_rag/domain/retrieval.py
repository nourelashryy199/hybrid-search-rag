from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hybrid_rag.domain.document import DocumentChunk


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Search score must be finite")

        return value


class SearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=5, gt=0)