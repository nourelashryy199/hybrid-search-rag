
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=20, ge=1, le=200)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty")
        return value


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    abstained: bool
