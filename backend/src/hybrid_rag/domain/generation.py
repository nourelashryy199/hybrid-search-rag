
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class EvidencePassage(BaseModel):
    """A retrieved passage supplied to an answer generator."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str
    page_number: int

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty")
        return value

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Page number must be positive")
        return value


class AnswerCitation(BaseModel):
    """Reference to a passage used as evidence."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    page_number: int

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Citation fields must not be empty")
        return value

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Page number must be positive")
        return value


class GeneratedAnswer(BaseModel):
    """Structured answer returned by a generation model."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[AnswerCitation, ...] = ()
    abstained: bool = False

    @model_validator(mode="after")
    def validate_answer(self):
        if not self.answer.strip():
            raise ValueError("Answer must not be empty")

        if self.abstained and self.citations:
            raise ValueError(
                "Abstained answers must not contain citations"
            )

        return self


class AnswerGenerator(Protocol):
    """Interface for evidence-grounded answer generation."""

    def generate(
        self,
        question: str,
        evidence: list[EvidencePassage],
    ) -> GeneratedAnswer:
        ...
