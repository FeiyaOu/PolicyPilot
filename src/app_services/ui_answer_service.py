from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.app_services.knowledge_base_loader import KnowledgeBaseLoadResult, KnowledgeBaseStatus
from src.app_services.retrieval_service import RetrievalMode
from src.generation.answer_contract import AnswerResult, AnswerStatus
from src.generation.answer_generator import AnswerProvider, generate_answer
from src.generation.evidence import evaluate_evidence


@dataclass(frozen=True)
class UiAnswerService:
    knowledge_base: KnowledgeBaseLoadResult
    answer_provider: AnswerProvider

    def answer(
        self,
        question: str,
        top_k: int = 4,
        min_score: float = 0.2,
        mode: RetrievalMode = RetrievalMode.BM25,
    ) -> AnswerResult:
        if self.knowledge_base.status != KnowledgeBaseStatus.READY or self.knowledge_base.retrieval_service is None:
            return _knowledge_base_fallback(question, self.knowledge_base.message, min_score)

        retrieval_results = self.knowledge_base.retrieval_service.search(question, mode=mode, top_k=top_k)
        review = evaluate_evidence(
            question=question,
            retrieval_results=retrieval_results,
            min_score=min_score,
            top_k=top_k,
        )
        return generate_answer(review, self.answer_provider)


def _knowledge_base_fallback(question: str, message: str, min_score: float) -> AnswerResult:
    return AnswerResult(
        question=question,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        answer_text=None,
        fallback_message=message,
        contexts=[],
        citations=[],
        retrieval_summary={
            "retrieved_count": 0,
            "selected_count": 0,
            "max_score": None,
            "min_score": min_score,
            "reason": "knowledge_base_missing",
        },
    )