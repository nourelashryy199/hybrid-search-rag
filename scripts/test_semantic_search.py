
from hybrid_rag.application.embedding_service import EmbeddingService
from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddingConfig
from hybrid_rag.infrastructure.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from hybrid_rag.infrastructure.vector_index import InMemoryVectorIndex


def main() -> None:
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
    )

    model = SentenceTransformerAdapter(config)
    embedding_service = EmbeddingService(model)

    chunks = [
        DocumentChunk(
            chunk_id="leave-001",
            document_id="employee-policy",
            text=(
                "Full-time employees receive 25 days "
                "of paid annual leave each year."
            ),
            chunk_index=0,
            page_number=1,
        ),
        DocumentChunk(
            chunk_id="remote-001",
            document_id="employee-policy",
            text=(
                "Employees may work remotely with "
                "prior approval from their manager."
            ),
            chunk_index=1,
            page_number=2,
        ),
        DocumentChunk(
            chunk_id="security-001",
            document_id="security-policy",
            text=(
                "All employees must use multi-factor "
                "authentication when accessing company systems."
            ),
            chunk_index=0,
            page_number=1,
        ),
        DocumentChunk(
            chunk_id="expenses-001",
            document_id="finance-policy",
            text=(
                "Business travel expenses must be submitted "
                "within 30 days of the trip."
            ),
            chunk_index=0,
            page_number=3,
        ),
    ]

    embedded_chunks = embedding_service.embed_chunks(chunks)

    index = InMemoryVectorIndex(dimensions=model.dimensions)
    index.add(chunks, embedded_chunks)

    question = "How much paid vacation time do staff get?"

    query_vector = model.embed_query(question)

    results = index.search(query_vector, top_k=3)

    print(f"\nQuestion: {question}")
    print("\nSemantic search results:\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {result.chunk.document_id}")
        print(f"Page: {result.chunk.page_number}")
        print(f"Text: {result.chunk.text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
