
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hybrid_rag.domain.embedding import EmbeddingConfig
from hybrid_rag.infrastructure.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)


@pytest.fixture
def mock_model():
    with patch(
        "hybrid_rag.infrastructure."
        "sentence_transformer_adapter.SentenceTransformer"
    ) as mock_constructor:
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 384
        model.encode.return_value = np.array(
            [[0.6, 0.8] + [0.0] * 382],
            dtype=float,
        )
        mock_constructor.return_value = model
        yield model


def make_adapter():
    return SentenceTransformerAdapter(
        EmbeddingConfig(
            model_name="test-model",
            dimensions=384,
        )
    )


def test_adapter_dimensions(mock_model):
    adapter = make_adapter()

    assert adapter.dimensions == 384


def test_embed_documents(mock_model):
    adapter = make_adapter()

    vectors = adapter.embed_documents(["Example text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384

    mock_model.encode.assert_called_once_with(
        ["Example text"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_query(mock_model):
    adapter = make_adapter()

    vector = adapter.embed_query("What is the policy?")

    assert len(vector) == 384


def test_empty_document_list(mock_model):
    adapter = make_adapter()

    assert adapter.embed_documents([]) == []

    mock_model.encode.assert_not_called()


def test_blank_query_is_rejected(mock_model):
    adapter = make_adapter()

    with pytest.raises(ValueError):
        adapter.embed_query("   ")


def test_blank_document_is_rejected(mock_model):
    adapter = make_adapter()

    with pytest.raises(ValueError):
        adapter.embed_documents(["Valid", "   "])


def test_dimension_mismatch(mock_model):
    with pytest.raises(
        ValueError,
        match="Configured dimensions",
    ):
        SentenceTransformerAdapter(
            EmbeddingConfig(
                model_name="test-model",
                dimensions=768,
            )
        )
