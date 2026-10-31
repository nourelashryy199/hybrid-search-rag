
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_type: str = Field(min_length=1)


class DocumentPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)


def generate_chunk_id(
    document_id: str,
    chunk_index: int,
    text: str,
) -> str:
    if not document_id:
        raise ValueError("document_id cannot be empty")

    if chunk_index < 0:
        raise ValueError("chunk_index cannot be negative")

    if not text:
        raise ValueError("text cannot be empty")

    payload = f"{document_id}:{chunk_index}:{text}"

    return sha256(payload.encode("utf-8")).hexdigest()
