
from typing import Protocol

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import (
    IndexedDocument,
    IndexingAction,
    determine_indexing_action,
)


class DocumentRepository(Protocol):
    def get(self, document_id: str) -> IndexedDocument | None:
        ...

    def save(self, document: IndexedDocument) -> None:
        ...


class ChunkEmbedder(Protocol):
    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        ...


class VectorStore(Protocol):
    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[EmbeddedChunk],
    ) -> None:
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        ...


class IndexingService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedder: ChunkEmbedder,
        vector_store: VectorStore,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store

    def index(
        self,
        identity: DocumentIdentity,
        chunks: list[DocumentChunk],
    ) -> IndexingAction:
        existing = self.repository.get(identity.document_id)

        action = determine_indexing_action(
            incoming=identity,
            existing=existing,
        )

        if action == IndexingAction.SKIP:
            return action

        if not chunks:
            raise ValueError("Cannot index a document without chunks")

        if any(
            chunk.document_id != identity.document_id
            for chunk in chunks
        ):
            raise ValueError(
                "All chunks must belong to the indexed document"
            )

        embeddings = self.embedder.embed_chunks(chunks)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count must match chunk count"
            )

        new_chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(set(new_chunk_ids)) != len(new_chunk_ids):
            raise ValueError("Duplicate chunk IDs are not allowed")

        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_id != embedding.chunk_id:
                raise ValueError(
                    "Embedding and chunk IDs must match"
                )

        # Add new vectors before removing stale ones.

        if existing is not None:
            reused_ids = set(existing.chunk_ids) & set(new_chunk_ids)

            if reused_ids:
                raise ValueError(
                    "Replacement cannot reuse existing chunk IDs; "
                    "generate version-specific IDs before indexing"
                )



        indexed_document = IndexedDocument(
            identity=identity,
            chunk_ids=tuple(new_chunk_ids),
        )

        # Stage new vectors while retaining the previous version.
        self.vector_store.add(chunks, embeddings)

        try:
            # Commit the new document version.
            self.repository.save(indexed_document)
        except Exception:
            # Best-effort cleanup of the staged vectors.
            try:
                self.vector_store.delete(new_chunk_ids)
            except Exception:
                pass
            raise

        # The new version is now recorded as active.
        if existing is not None:
            stale_ids = sorted(
                set(existing.chunk_ids) - set(new_chunk_ids)
            )

            if stale_ids:
                try:
                    self.vector_store.delete(stale_ids)
                except Exception:
                    # The new version remains committed.
                    # Stale-vector cleanup will require a retry mechanism.
                    pass

        return action
