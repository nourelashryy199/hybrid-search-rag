
from sentence_transformers import SentenceTransformer

from hybrid_rag.domain.embedding import EmbeddingConfig


class SentenceTransformerAdapter:
    def __init__(
        self,
        config: EmbeddingConfig,
    ) -> None:
        self.config = config
        self.model = SentenceTransformer(config.model_name)

        actual_dimensions = (
            self.model.get_sentence_embedding_dimension()
        )

        if actual_dimensions != config.dimensions:
            raise ValueError(
                f"Configured dimensions: {config.dimensions}; "
                f"model dimensions: {actual_dimensions}"
            )

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Document texts cannot be empty"
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Query text cannot be empty"
            )

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding[0].tolist()
