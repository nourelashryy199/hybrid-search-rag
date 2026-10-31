
from math import inf, nan

import pytest
from pydantic import ValidationError

from hybrid_rag.domain.embedding import (
    EmbeddedChunk,
    EmbeddingConfig,
)


def test_embedding_config():
    config = EmbeddingConfig(
        model_name="example-model",
        dimensions=384,
    )

    assert config.dimensions == 384


def test_embedding_config_rejects_invalid_dimensions():
    with pytest.raises(ValidationError):
        EmbeddingConfig(
            model_name="example-model",
            dimensions=0,
        )


def test_embedded_chunk():
    embedded = EmbeddedChunk(
        chunk_id="chunk-001",
        vector=(0.1, 0.2, 0.3),
    )

    assert embedded.chunk_id == "chunk-001"
    assert len(embedded.vector) == 3


def test_embedding_rejects_empty_vector():
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk_id="chunk-001",
            vector=(),
        )


@pytest.mark.parametrize("invalid_value", [nan, inf, -inf])
def test_embedding_rejects_nonfinite_values(invalid_value):
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk_id="chunk-001",
            vector=(0.1, invalid_value),
        )
