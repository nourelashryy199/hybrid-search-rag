
from hybrid_rag.application.factory import create_rag_application
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.domain.generation import (
    AnswerCitation,
    GeneratedAnswer,
)
from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.infrastructure.config import AppSettings



class FakeEmbeddingModel:
    @property
    def dimensions(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        vectors = []

        for text in texts:
            lowered = text.lower()

            vectors.append([
                float("leave" in lowered),
                float("security" in lowered),
                float("expense" in lowered),
            ])

        return vectors


    def embed_query(self, text: str) -> list[float]:
        vector = self.embed_documents([text])[0]

        if not any(vector):
            return [1.0, 0.0, 0.0]

        return vector


class FakeAnswerGenerator:
    def generate(self, question, evidence):
        if not evidence:
            return GeneratedAnswer(
                answer="Insufficient evidence.",
                abstained=True,
            )

        passage = evidence[0]

        return GeneratedAnswer(
            answer=passage.text,
            citations=(
                AnswerCitation(
                    chunk_id=passage.chunk_id,
                    document_id=passage.document_id,
                    page_number=passage.page_number,
                ),
            ),
        )


def make_settings(tmp_path):
    return AppSettings(
        qdrant_path=tmp_path / "qdrant",
        sqlite_path=tmp_path / "documents.db",
        embedding_dimensions=3,
        enable_reranking=False,
    )


def make_document():
    identity = DocumentIdentity(
        document_id="handbook",
        content_hash="a" * 64,
    )

    chunks = [
        DocumentChunk(
            chunk_id="leave-001",
            document_id="handbook",
            text="Employees receive 25 days of annual leave.",
            chunk_index=0,
            page_number=4,
        ),
        DocumentChunk(
            chunk_id="security-001",
            document_id="handbook",
            text="Security requires multifactor authentication.",
            chunk_index=1,
            page_number=5,
        ),
    ]

    return identity, chunks


def test_factory_creates_retrieval_application(tmp_path):
    settings = make_settings(tmp_path)

    with create_rag_application(
        settings,
        embedding_model=FakeEmbeddingModel(),
    ) as app:
        assert app.repository is not None
        assert app.vector_store is not None
        assert app.indexing is not None
        assert app.retrieval is not None
        assert app.rag is None


def test_factory_supports_indexing_and_retrieval(tmp_path):
    settings = make_settings(tmp_path)

    with create_rag_application(
        settings,
        embedding_model=FakeEmbeddingModel(),
    ) as app:
        identity, chunks = make_document()

        app.indexing.index(identity, chunks)

        results = app.retrieval.search(
            "annual leave",
            top_k=2,
            candidate_k=2,
        )

        assert results
        assert results[0].chunk.chunk_id == "leave-001"


def test_factory_supports_grounded_generation(tmp_path):
    settings = make_settings(tmp_path)

    with create_rag_application(
        settings,
        embedding_model=FakeEmbeddingModel(),
        answer_generator=FakeAnswerGenerator(),
    ) as app:
        identity, chunks = make_document()
        app.indexing.index(identity, chunks)

        assert app.rag is not None

        answer = app.rag.ask(
            "annual leave",
            top_k=2,
            candidate_k=2,
        )

        assert answer.abstained is False
        assert answer.citations
        assert answer.citations[0].chunk_id == "leave-001"


def test_factory_recovers_index_after_restart(tmp_path):
    settings = make_settings(tmp_path)

    identity, chunks = make_document()

    with create_rag_application(
        settings,
        embedding_model=FakeEmbeddingModel(),
    ) as app:
        app.indexing.index(identity, chunks)

    with create_rag_application(
        settings,
        embedding_model=FakeEmbeddingModel(),
    ) as reopened:
        results = reopened.retrieval.search(
            "annual leave",
            top_k=2,
            candidate_k=2,
        )

        assert results
        assert results[0].chunk.chunk_id == "leave-001"


def test_factory_abstains_when_no_documents_exist(tmp_path):
    settings = make_settings(tmp_path)

    with create_rag_application(
        settings,
        embedding_model=FakeEmbeddingModel(),
        answer_generator=FakeAnswerGenerator(),
    ) as app:
        answer = app.rag.ask(
            "What is the vacation policy?",
            top_k=2,
            candidate_k=2,
        )

        assert answer.abstained is True
        assert answer.citations == ()
