
from collections.abc import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty")

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[set[str]],
) -> float:
    if len(rankings) != len(relevant_sets):
        raise ValueError(
            "Rankings and relevance sets must have equal lengths"
        )

    if not rankings:
        raise ValueError("Evaluation dataset cannot be empty")

    return sum(
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_sets)
    ) / len(rankings)
