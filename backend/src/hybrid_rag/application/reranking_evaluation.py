
from dataclasses import dataclass
from typing import Protocol

from hybrid_rag.domain.reranking import RerankedResult


class RetrievalSystem(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RerankedResult]:
        ...


@dataclass(frozen=True)
class EvaluationQuery:
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    mrr: float


@dataclass(frozen=True)
class EvaluationComparison:
    baseline: RetrievalMetrics
    reranked: RetrievalMetrics

    @property
    def recall_at_1_delta(self) -> float:
        return (
            self.reranked.recall_at_1
            - self.baseline.recall_at_1
        )

    @property
    def recall_at_3_delta(self) -> float:
        return (
            self.reranked.recall_at_3
            - self.baseline.recall_at_3
        )

    @property
    def mrr_delta(self) -> float:
        return self.reranked.mrr - self.baseline.mrr


def evaluate_retrieval(
    system: RetrievalSystem,
    queries: list[EvaluationQuery],
    candidate_k: int = 20,
) -> RetrievalMetrics:
    if not queries:
        raise ValueError("Evaluation queries must not be empty")

    if candidate_k < 3:
        raise ValueError("candidate_k must be at least 3")

    recall_1_total = 0.0
    recall_3_total = 0.0
    reciprocal_rank_total = 0.0

    for item in queries:
        if not item.query.strip():
            raise ValueError("Evaluation query must not be empty")

        if not item.relevant_chunk_ids:
            raise ValueError(
                "Each query must have relevant chunk IDs"
            )

        results = system.search(
            query=item.query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        retrieved_ids = [
            result.chunk.chunk_id
            for result in results
        ]

        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError(
                "Retrieval results contain duplicate chunk IDs"
            )

        if len(retrieved_ids) > candidate_k:
            raise ValueError(
                "Retrieval returned more than candidate_k results"
            )

        relevant = item.relevant_chunk_ids

        # Hit-based Recall@k, matching our earlier
        # single-relevant-chunk benchmark.
        recall_1_total += float(
            any(chunk_id in relevant for chunk_id in retrieved_ids[:1])
        )

        recall_3_total += float(
            any(chunk_id in relevant for chunk_id in retrieved_ids[:3])
        )

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant:
                reciprocal_rank_total += 1.0 / rank
                break

    count = len(queries)

    return RetrievalMetrics(
        recall_at_1=recall_1_total / count,
        recall_at_3=recall_3_total / count,
        mrr=reciprocal_rank_total / count,
    )


def compare_retrieval_systems(
    baseline: RetrievalSystem,
    reranked: RetrievalSystem,
    queries: list[EvaluationQuery],
    candidate_k: int = 20,
) -> EvaluationComparison:
    baseline_metrics = evaluate_retrieval(
        baseline,
        queries,
        candidate_k,
    )

    reranked_metrics = evaluate_retrieval(
        reranked,
        queries,
        candidate_k,
    )

    return EvaluationComparison(
        baseline=baseline_metrics,
        reranked=reranked_metrics,
    )
