from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from src.generation.citations import Citation
from src.generation.evidence import EvidenceReview


class AnswerStatus(StrEnum):
    ANSWER_READY = "answer_ready"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERATION_FAILED = "generation_failed"


@dataclass(frozen=True)
class AnswerResult:
    question: str
    status: AnswerStatus
    answer_text: str | None
    fallback_message: str | None
    contexts: list[dict[str, Any]]
    citations: list[Citation]
    retrieval_summary: dict[str, Any]


def build_answer_output(review: EvidenceReview, answer_text: str | None = None) -> AnswerResult:
    if not review.is_sufficient:
        return AnswerResult(
            question=review.question,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            answer_text=None,
            fallback_message="未检索到足够可靠的政策依据，暂时无法回答该问题。",
            contexts=[],
            citations=[],
            retrieval_summary=review.retrieval_summary,
        )

    return AnswerResult(
        question=review.question,
        status=AnswerStatus.ANSWER_READY,
        answer_text=answer_text,
        fallback_message=None,
        contexts=[
            {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "source_file": result.source_file,
                "page_number": result.page_number,
                "score": result.fused_score,
            }
            for result in review.selected_results
        ],
        citations=review.citations,
        retrieval_summary=review.retrieval_summary,
    )


def build_generation_fallback_output(review: EvidenceReview, fallback_message: str, reason: str) -> AnswerResult:
    answer_output = build_answer_output(review)
    return replace(
        answer_output,
        status=AnswerStatus.GENERATION_FAILED,
        answer_text=None,
        fallback_message=fallback_message,
        retrieval_summary={
            **answer_output.retrieval_summary,
            "generation_status": "failed",
            "generation_reason": reason,
        },
    )