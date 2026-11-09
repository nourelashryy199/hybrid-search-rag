
from dataclasses import dataclass

from hybrid_rag.application.bm25_recovery import BM25RecoveryService
from hybrid_rag.application.embedding_service import EmbeddingService
from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.grounded_answer import GroundedAnswerService
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.application.indexing_service import IndexingService
from hybrid_rag.application.live_keyword_retrieval import LiveKeywordRetriever
from hybrid_rag.application.rag_service import RAGService
from hybrid_rag.application.reranking_service import RerankingService
from hybrid_rag.application.retrieval_pipeline import RetrievalPipeline
from hybrid_rag.application.synchronized_indexing import (
    SynchronizedIndexingService,
)
from hybrid_rag.application.version_filtered_keyword_retrieval import (
    VersionFilteredKeywordRetriever,
)
from hybrid_rag.application.version_filtered_retrieval import (
    VersionFilteredVectorRetriever,
)

from hybrid_rag.domain.embedding import EmbeddingConfig
from hybrid_rag.infrastructure.claude_generator import ClaudeGenerator
from hybrid_rag.infrastructure.cohere_reranker import CohereReranker
from hybrid_rag.infrastructure.config import AppSettings
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore
from hybrid_rag.infrastructure.runtime import ApplicationRuntime
from hybrid_rag.infrastructure.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from hybrid_rag.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


@dataclass
class RAGApplication:
    settings: AppSettings
    runtime: ApplicationRuntime
    repository: SQLiteDocumentRepository
    vector_store: QdrantVectorStore
    indexing: SynchronizedIndexingService
    retrieval: RetrievalPipeline
    rag: RAGService | None

    def close(self) -> None:
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def create_rag_application(
    settings: AppSettings,
    embedding_model=None,
    answer_generator=None,
    reranker=None,
) -> RAGApplication:
    """
    Assemble persistent indexing, retrieval, and optional generation.

    Injectable components allow offline integration tests.
    """

    runtime = ApplicationRuntime()

    try:
        settings.qdrant_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        settings.sqlite_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        repository = SQLiteDocumentRepository(
            settings.sqlite_path
        )

        vector_store = QdrantVectorStore(
            storage_path=settings.qdrant_path,
            collection_name=settings.qdrant_collection,
            dimensions=settings.embedding_dimensions,
        )

        runtime.register(vector_store)

        if embedding_model is None:
            embedding_model = SentenceTransformerAdapter(
                EmbeddingConfig(
                    model_name=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                )
            )

        embedder = EmbeddingService(embedding_model)

        indexer = IndexingService(
            repository=repository,
            embedder=embedder,
            vector_store=vector_store,
        )

        recovery = BM25RecoveryService(
            repository=repository,
            chunk_reader=vector_store,
        )

        synchronized_indexer = SynchronizedIndexingService(
            indexer=indexer,
            keyword_index_builder=recovery,
        )

        vector_retriever = VersionFilteredVectorRetriever(
            retriever=vector_store,
            repository=repository,
        )

        keyword_retriever = VersionFilteredKeywordRetriever(
            retriever=LiveKeywordRetriever(
                synchronized_indexer
            ),
            repository=repository,
        )

        hybrid = HybridRetrievalService(
            embedding_model=embedding_model,
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            fusion=ReciprocalRankFusion(),
        )

        if reranker is None and settings.enable_reranking:
            import cohere

            cohere_client = cohere.ClientV2(
                api_key=settings.cohere_api_key
            )

            reranker = CohereReranker(
                client=cohere_client,
                model=settings.cohere_model,
            )

        reranking_service = (
            RerankingService(reranker)
            if reranker is not None
            else None
        )

        retrieval = RetrievalPipeline(
            retriever=hybrid,
            reranker=reranking_service,
        )

        if answer_generator is None and settings.anthropic_api_key:
            if not settings.anthropic_model:
                raise ValueError(
                    "Anthropic model must be configured "
                    "when using the Anthropic API"
                )

            import anthropic

            anthropic_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key
            )

            answer_generator = ClaudeGenerator(
                client=anthropic_client,
                model=settings.anthropic_model,
            )

        rag = None

        if answer_generator is not None:
            rag = RAGService(
                retrieval_pipeline=retrieval,
                answer_service=GroundedAnswerService(
                    answer_generator
                ),
            )

        return RAGApplication(
            settings=settings,
            runtime=runtime,
            repository=repository,
            vector_store=vector_store,
            indexing=synchronized_indexer,
            retrieval=retrieval,
            rag=rag,
        )

    except BaseException:
        runtime.close()
        raise
