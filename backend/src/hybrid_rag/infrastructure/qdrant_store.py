from pathlib import Path

from qdrant_client import QdrantClient, models
from uuid import NAMESPACE_URL, uuid5
import math

from hybrid_rag.domain.retrieval import SearchResult

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk


class QdrantVectorStore:
    """Manage a persistent Qdrant collection for document embeddings."""

    def __init__(
        self,
        storage_path: str | Path,
        collection_name: str = "hybrid_rag_chunks",
        dimensions: int = 384,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.dimensions = dimensions

        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=str(self.storage_path))

        try:
            self._ensure_collection()
        except Exception:
            self.client.close()
            raise

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection = self.client.get_collection(self.collection_name)

        vector_config = collection.config.params.vectors

        if not isinstance(vector_config, models.VectorParams):
            raise ValueError(
                "Expected a single unnamed dense vector configuration"
            )

        if vector_config.size != self.dimensions:
            raise ValueError(
                f"Collection dimension mismatch: "
                f"expected {self.dimensions}, "
                f"found {vector_config.size}"
            )

        if vector_config.distance != models.Distance.COSINE:
            raise ValueError(
                "Existing collection must use cosine distance"
            )

    def close(self) -> None:
        """Release the Qdrant client and local storage lock."""
        self.client.close()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Map a chunk ID deterministically to a Qdrant-compatible UUID."""
        return str(uuid5(NAMESPACE_URL, f"hybrid-rag:chunk:{chunk_id}"))

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[EmbeddedChunk],
    ) -> None:
        """Insert or update chunks and their vectors in Qdrant."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )

        if not chunks:
            return

        points = []
        seen_ids: set[str] = set()

        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_id != embedding.chunk_id:
                raise ValueError(
                    "Chunk and embedding IDs must match"
                )

            if chunk.chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk.chunk_id}"
                )

            seen_ids.add(chunk.chunk_id)

            if len(embedding.vector) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"received {len(embedding.vector)}"
                )

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=list(embedding.vector),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete vectors by their original chunk IDs."""

        if not chunk_ids:
            return

        point_ids = [
            self._point_id(chunk_id)
            for chunk_id in set(chunk_ids)
        ]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=point_ids,
            ),
            wait=True,
        )
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve the most similar document chunks from Qdrant."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} query dimensions, "
                f"received {len(query_vector)}"
            )

        if not all(math.isfinite(value) for value in query_vector):
            raise ValueError("Query vector must contain finite values")

        if not any(value != 0 for value in query_vector):
            raise ValueError("Query vector cannot be zero")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload

            if payload is None:
                raise ValueError(
                    f"Missing payload for Qdrant point {point.id}"
                )

            try:
                chunk = DocumentChunk(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    text=payload["text"],
                    chunk_index=payload["chunk_index"],
                    page_number=payload["page_number"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid payload for Qdrant point {point.id}"
                ) from exc

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(point.score),
                )
            )

        return sorted(
            results,
            key=lambda result: (
                -result.score,
                result.chunk.chunk_id,
            ),
        )


    def get_chunks(
        self,
        chunk_ids: list[str],
    ) -> list[DocumentChunk]:
        """Retrieve stored chunks by ID, preserving requested order."""

        if not chunk_ids:
            return []

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs are not allowed")

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._point_id(chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
            with_vectors=False,
        )

        chunks_by_id: dict[str, DocumentChunk] = {}

        for point in points:
            payload = point.payload

            if payload is None:
                raise ValueError(
                    f"Missing payload for Qdrant point {point.id}"
                )

            try:
                chunk = DocumentChunk(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    text=payload["text"],
                    chunk_index=payload["chunk_index"],
                    page_number=payload["page_number"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid payload for Qdrant point {point.id}"
                ) from exc

            expected_point_id = self._point_id(chunk.chunk_id)

            if str(point.id) != expected_point_id:
                raise ValueError(
                    f"Point ID does not match chunk ID: {chunk.chunk_id}"
                )

            if chunk.chunk_id in chunks_by_id:
                raise ValueError(
                    f"Duplicate retrieved chunk: {chunk.chunk_id}"
                )

            chunks_by_id[chunk.chunk_id] = chunk

        missing_ids = [
            chunk_id
            for chunk_id in chunk_ids
            if chunk_id not in chunks_by_id
        ]

        if missing_ids:
            raise ValueError(
                f"Missing active chunks in Qdrant: {missing_ids}"
            )

        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids]
