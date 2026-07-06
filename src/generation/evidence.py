from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.app_services.retrieval_service import RetrievalResult
from src.generation.citations import Citation, build_citations


class EvidenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class EvidenceReview:
    question: str
    status: EvidenceStatus
    is_sufficient: bool
    selected_results: list[RetrievalResult]
    citations: list[Citation]
    retrieval_summary: dict[str, Any]


def evaluate_evidence(
    question: str,
    retrieval_results: list[RetrievalResult],
    min_score: float = 0.2,
    top_k: int = 4,
) -> EvidenceReview:
    if not retrieval_results:
        return _insufficient_review(
            question=question,
            retrieved_count=0,
            max_score=None,
            min_score=min_score,
            reason="no_retrieval_results",
        )

    sorted_results = sorted(retrieval_results, key=lambda result: result.fused_score, reverse=True)
    max_score = sorted_results[0].fused_score

    if max_score < min_score:
        return _insufficient_review(
            question=question,
            retrieved_count=len(retrieval_results),
            max_score=max_score,
            min_score=min_score,
            reason="score_below_threshold",
        )

    selected_results = sorted_results[:top_k]

    return EvidenceReview(
        question=question,
        status=EvidenceStatus.SUFFICIENT,
        is_sufficient=True,
        selected_results=selected_results,
        citations=build_citations(selected_results),
        retrieval_summary={
            "retrieved_count": len(retrieval_results),
            "selected_count": len(selected_results),
            "max_score": max_score,
            "min_score": min_score,
            "reason": "sufficient_evidence",
        },
    )


def _insufficient_review(
    question: str,
    retrieved_count: int,
    max_score: float | None,
    min_score: float,
    reason: str,
) -> EvidenceReview:
    return EvidenceReview(
        question=question,
        status=EvidenceStatus.INSUFFICIENT,
        is_sufficient=False,
        selected_results=[],
        citations=[],
        retrieval_summary={
            "retrieved_count": retrieved_count,
            "selected_count": 0,
            "max_score": max_score,
            "min_score": min_score,
            "reason": reason,
        },
    )