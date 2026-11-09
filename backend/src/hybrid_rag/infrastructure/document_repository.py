
from hybrid_rag.domain.indexing import IndexedDocument


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, IndexedDocument] = {}

    def get(self, document_id: str) -> IndexedDocument | None:
        return self._documents.get(document_id)

    def save(self, document: IndexedDocument) -> None:
        self._documents[document.identity.document_id] = document

    def delete(self, document_id: str) -> None:
        self._documents.pop(document_id, None)

    def list_all(self) -> list[IndexedDocument]:
        return list(self._documents.values())
