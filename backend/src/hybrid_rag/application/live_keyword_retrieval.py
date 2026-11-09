from typing import Protocol
from hybrid_rag.domain.retrieval import SearchResult
from hybrid_rag.infrastructure.bm25_index import BM25Index

class KeywordIndexProvider(Protocol):
    @property
    def keyword_index(self) -> BM25Index:
        ...
class LiveKeywordRetriever:
    def __init__(
        self,
        provider: KeywordIndexProvider,
    ) -> None:
        self.provider = provider

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        return self.provider.keyword_index.search(
            query,
            top_k=top_k,
        )
