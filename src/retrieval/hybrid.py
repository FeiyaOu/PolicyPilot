from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    vector_score: float
    bm25_score: float
    fused_score: float


def fuse_hybrid_scores(
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    alpha: float = 0.5,
) -> list[HybridSearchResult]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    chunk_ids = set(vector_scores) | set(bm25_scores)
    results = []

    for chunk_id in chunk_ids:
        vector_score = vector_scores.get(chunk_id, 0.0)
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        fused_score = alpha * vector_score + (1 - alpha) * bm25_score
        results.append(
            HybridSearchResult(
                chunk_id=chunk_id,
                vector_score=vector_score,
                bm25_score=bm25_score,
                fused_score=fused_score,
            )
        )

    return sorted(results, key=lambda result: result.fused_score, reverse=True)
