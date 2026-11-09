
from typing import Protocol

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.indexing import IndexedDocument
from hybrid_rag.infrastructure.bm25_index import BM25Index


class DocumentCatalog(Protocol):
    def list_all(self) -> list[IndexedDocument]:
        ...


class ChunkReader(Protocol):
    def get_chunks(
        self,
        chunk_ids: list[str],
    ) -> list[DocumentChunk]:
        ...


class BM25RecoveryService:
    """Reconstruct a BM25 index from active persisted chunks."""

    def __init__(
        self,
        repository: DocumentCatalog,
        chunk_reader: ChunkReader,
    ) -> None:
        self.repository = repository
        self.chunk_reader = chunk_reader

    def rebuild(self) -> BM25Index:
        """Return a fresh BM25 index containing only active chunks."""

        documents = self.repository.list_all()

        active_chunk_ids: list[str] = []
        expected_document_ids: dict[str, str] = {}

        for document in documents:
            document_id = document.identity.document_id

            for chunk_id in document.chunk_ids:
                if chunk_id in expected_document_ids:
                    raise ValueError(
                        f"Duplicate active chunk ID: {chunk_id}"
                    )

                expected_document_ids[chunk_id] = document_id
                active_chunk_ids.append(chunk_id)

        chunks = self.chunk_reader.get_chunks(active_chunk_ids)

        if len(chunks) != len(active_chunk_ids):
            raise ValueError(
                "Recovered chunk count does not match active metadata"
            )

        for expected_id, chunk in zip(active_chunk_ids, chunks):
            if chunk.chunk_id != expected_id:
                raise ValueError(
                    f"Recovered chunk order mismatch: {expected_id}"
                )

            if chunk.document_id != expected_document_ids[expected_id]:
                raise ValueError(
                    f"Recovered chunk belongs to wrong document: "
                    f"{chunk.chunk_id}"
                )

        index = BM25Index()

        if chunks:
            index.add(chunks)

        return index
