
from hybrid_rag.application.embedding_service import EmbeddingService
from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.domain.embedding import EmbeddingConfig
from hybrid_rag.infrastructure.bm25_index import BM25Index
from hybrid_rag.infrastructure.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from hybrid_rag.infrastructure.vector_index import InMemoryVectorIndex

from evaluation.run_retrieval_evaluation import load_queries
from evaluation.sample_corpus import get_sample_chunks


def print_rankings(name, results, relevant_ids):
    print(f"\n{name}")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):
        chunk_id = result.chunk.chunk_id
        marker = "*" if chunk_id in relevant_ids else " "

        print(
            f"{marker} Rank {rank}: "
            f"{chunk_id:<15} "
            f"Score: {result.score:.6f}"
        )


def main():
    chunks = get_sample_chunks()
    queries = load_queries()

    model = SentenceTransformerAdapter(
        EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
        )
    )

    embedded = EmbeddingService(model).embed_chunks(chunks)

    vector_index = InMemoryVectorIndex(
        dimensions=model.dimensions
    )
    vector_index.add(chunks, embedded)

    bm25_index = BM25Index()
    bm25_index.add(chunks)

    hybrid = HybridRetrievalService(
        embedding_model=model,
        vector_retriever=vector_index,
        keyword_retriever=bm25_index,
        fusion=ReciprocalRankFusion(k=60),
    )

    selected_ids = {"q001", "q004", "q011", "q012"}

    for item in queries:
        if item["query_id"] not in selected_ids:
            continue

        question = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        dense = vector_index.search(
            model.embed_query(question),
            top_k=5,
        )

        keyword = bm25_index.search(
            question,
            top_k=5,
        )

        fused = hybrid.search(
            question,
            top_k=5,
            candidate_k=5,
        )

        print("\n" + "=" * 70)
        print(f"{item['query_id']}: {question}")
        print(f"Relevant: {sorted(relevant_ids)}")

        print_rankings("DENSE", dense, relevant_ids)
        print_rankings("BM25", keyword, relevant_ids)
        print_rankings("HYBRID", fused, relevant_ids)


if __name__ == "__main__":
    main()
