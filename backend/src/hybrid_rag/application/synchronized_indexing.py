
from typing import Protocol

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexingAction
from hybrid_rag.infrastructure.bm25_index import BM25Index


class DocumentIndexer(Protocol):
    def index(
        self,
        identity: DocumentIdentity,
        chunks: list[DocumentChunk],
    ) -> IndexingAction:
        ...


class KeywordIndexBuilder(Protocol):
    def rebuild(self) -> BM25Index:
        ...


class SynchronizedIndexingService:
    """Index documents and refresh the active lexical index."""

    def __init__(
        self,
        indexer: DocumentIndexer,
        keyword_index_builder: KeywordIndexBuilder,
    ) -> None:
        self.indexer = indexer
        self.keyword_index_builder = keyword_index_builder

        # Reconstruct the lexical index from persistent storage.
        self._keyword_index = self.keyword_index_builder.rebuild()

    @property
    def keyword_index(self) -> BM25Index:
        return self._keyword_index

    def index(
        self,
        identity: DocumentIdentity,
        chunks: list[DocumentChunk],
    ) -> IndexingAction:
        """Index a document and refresh BM25 after a change."""

        action = self.indexer.index(identity, chunks)

        if action != IndexingAction.SKIP:
            # Build first, then publish the replacement index.
            refreshed_index = self.keyword_index_builder.rebuild()
            self._keyword_index = refreshed_index

        return action
