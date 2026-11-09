
import json
from pathlib import Path

from hybrid_rag.application.embedding_service import EmbeddingService
from hybrid_rag.application.evaluation import (
    mean_reciprocal_rank,
    recall_at_k,
)
from hybrid_rag.application.fusion import ReciprocalRankFusion
from hybrid_rag.application.hybrid_retrieval import HybridRetrievalService
from hybrid_rag.domain.embedding import EmbeddingConfig
from hybrid_rag.infrastructure.bm25_index import BM25Index
from hybrid_rag.infrastructure.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from hybrid_rag.infrastructure.vector_index import InMemoryVectorIndex

from evaluation.sample_corpus import get_sample_chunks


EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "retrieval_queries.json"


def load_queries() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as file:
        queries = json.load(file)

    if not queries:
        raise ValueError("Evaluation dataset cannot be empty")

    corpus_ids = {
        chunk.chunk_id
        for chunk in get_sample_chunks()
    }

    for item in queries:
        if not item["query"].strip():
            raise ValueError("Evaluation query cannot be empty")

        relevant_ids = set(item["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError(
                "Every query must have relevant chunk IDs"
            )

        if not relevant_ids.issubset(corpus_ids):
            raise ValueError(
                f"Unknown relevant chunk ID in {item['query_id']}"
            )

    return queries


def evaluate_rankings(
    rankings: list[list[str]],
    relevant_sets: list[set[str]],
) -> dict[str, float]:
    count = len(rankings)

    return {
        "Recall@1": sum(
            recall_at_k(ranking, relevant, 1)
            for ranking, relevant in zip(rankings, relevant_sets)
        ) / count,
        "Recall@3": sum(
            recall_at_k(ranking, relevant, 3)
            for ranking, relevant in zip(rankings, relevant_sets)
        ) / count,
        "MRR": mean_reciprocal_rank(
            rankings,
            relevant_sets,
        ),
    }


def main() -> None:
    queries = load_queries()
    chunks = get_sample_chunks()

    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
    )

    model = SentenceTransformerAdapter(config)

    embedded_chunks = EmbeddingService(
        model
    ).embed_chunks(chunks)

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

    methods = {
        "Dense": [],
        "BM25": [],
        "Hybrid": [],
    }

    relevant_sets = []

    for item in queries:
        query = item["query"]
        relevant_sets.append(
            set(item["relevant_chunk_ids"])
        )

        query_vector = model.embed_query(query)

        dense_results = vector_index.search(
            query_vector,
            top_k=len(chunks),
        )

        bm25_results = bm25_index.search(
            query,
            top_k=len(chunks),
        )

        hybrid_results = hybrid_service.search(
            query,
            top_k=len(chunks),
            candidate_k=len(chunks),
        )

        methods["Dense"].append([
            result.chunk.chunk_id
            for result in dense_results
        ])

        methods["BM25"].append([
            result.chunk.chunk_id
            for result in bm25_results
        ])

        methods["Hybrid"].append([
            result.chunk.chunk_id
            for result in hybrid_results
        ])

    print("\nRETRIEVAL EVALUATION")
    print("=" * 58)
    print(
        f"{'Method':<12}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'MRR':>12}"
    )
    print("-" * 58)

    for method, rankings in methods.items():
        metrics = evaluate_rankings(
            rankings,
            relevant_sets,
        )

        print(
            f"{method:<12}"
            f"{metrics['Recall@1']:>12.3f}"
            f"{metrics['Recall@3']:>12.3f}"
            f"{metrics['MRR']:>12.3f}"
        )


    print("=" * 58)

    print("\nPER-QUERY TOP RESULTS")
    print("=" * 90)

    for index, item in enumerate(queries):
        print(f"\n{item['query_id']}: {item['query']}")
        print(
            "Expected:",
            ", ".join(item["relevant_chunk_ids"]),
        )

        for method, rankings in methods.items():
            retrieved = rankings[index]
            top_result = retrieved[0] if retrieved else "NONE"

            correct = top_result in set(
                item["relevant_chunk_ids"]
            )

            status = "CORRECT" if correct else "INCORRECT"

            print(
                f"  {method:<8} "
                f"Top: {top_result:<15} "
                f"{status}"
            )


if __name__ == "__main__":
    main()
