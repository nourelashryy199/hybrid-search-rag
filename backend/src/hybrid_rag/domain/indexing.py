
from dataclasses import dataclass
from enum import Enum

from hybrid_rag.domain.identity import DocumentIdentity


class IndexingAction(str, Enum):
    CREATE = "create"
    SKIP = "skip"
    REPLACE = "replace"


@dataclass(frozen=True)
class IndexedDocument:
    identity: DocumentIdentity
    chunk_ids: tuple[str, ...]


def determine_indexing_action(
    incoming: DocumentIdentity,
    existing: IndexedDocument | None,
) -> IndexingAction:
    if existing is None:
        return IndexingAction.CREATE

    if incoming.document_id != existing.identity.document_id:
        raise ValueError(
            "Cannot compare different logical documents"
        )

    if incoming.content_hash == existing.identity.content_hash:
        return IndexingAction.SKIP

    return IndexingAction.REPLACE
