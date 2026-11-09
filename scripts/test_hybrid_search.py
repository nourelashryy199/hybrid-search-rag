
from hybrid_rag.application.embedding_service import EmbeddingService
from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddingConfig
from hybrid_rag.infrastructure.bm25_index import BM25Index
from hybrid_rag.infrastructure.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from hybrid_rag.infrastructure.vector_index import InMemoryVectorIndex
from evaluation.sample_corpus import get_sample_chunks


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    page_number: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        page_number=page_number,
    )


def main() -> None:
    chunks = get_sample_chunks()
    # chunks = [
    #     make_chunk(
    #         "leave-001",
    #         "employee-handbook",
    #         "Full-time employees receive 25 days of paid "
    #         "annual leave per year.",
    #         1,
    #     ),
    #     make_chunk(
    #         "leave-002",
    #         "employee-handbook",
    #         "Annual leave requests must be approved by "
    #         "the employee's manager.",
    #         2,
    #     ),
    #     make_chunk(
    #         "remote-001",
    #         "employee-handbook",
    #         "Employees may work remotely up to three "
    #         "days per week with manager approval.",
    #         3,
    #     ),
    #     make_chunk(
    #         "security-001",
    #         "security-policy",
    #         "Multi-factor authentication is required "
    #         "for all company accounts.",
    #         1,
    #     ),
    #     make_chunk(
    #         "expenses-001",
    #         "finance-policy",
    #         "Travel expenses must be submitted within "
    #         "30 days of returning from a business trip.",
    #         4,
    #     ),
    #     make_chunk(
    #         "training-001",
    #         "employee-handbook",
    #         "Employees can request funding for "
    #         "professional development courses.",
    #         5,
    #     ),
    # ]

    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
    )

    model = SentenceTransformerAdapter(config)

    embedding_service = EmbeddingService(model)
    embedded_chunks = embedding_service.embed_chunks(chunks)

    vector_index = InMemoryVectorIndex(
        dimensions=model.dimensions
    )
    vector_index.add(chunks, embedded_chunks)

    bm25_index = BM25Index()
    bm25_index.add(chunks)

    hybrid_service = HybridRetrievalService(
        embedding_model=model,
        vector_retriever=vector_index,
        keyword_retriever=bm25_index,
        fusion=ReciprocalRankFusion(k=60),
    )

    questions = [
        "How many vacation days do employees get?",
        "Who approves annual leave requests?",
        "Is two-factor authentication required?",
    ]

    for question in questions:
        print(f"\nQUESTION: {question}")
        print("=" * 65)

        results = hybrid_service.search(
            query=question,
            top_k=3,
            candidate_k=6,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nRank {rank}")
            print(f"RRF score: {result.score:.6f}")
            print(f"Document: {result.chunk.document_id}")
            print(f"Page: {result.chunk.page_number}")
            print(f"Text: {result.chunk.text}")

        print("\n" + "-" * 65)


if __name__ == "__main__":
    main()
