from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.generation.answer_contract import AnswerResult, build_answer_output, build_generation_fallback_output
from src.generation.evidence import EvidenceReview


PROVIDER_ERROR_FALLBACK = "模型回答生成失败，请稍后重试或切换本地演示回答。"
EMPTY_ANSWER_FALLBACK = "模型返回为空，已拒绝生成无依据答案。"


@dataclass(frozen=True)
class AnswerGenerationInput:
    question: str
    contexts: list[dict[str, Any]]


class AnswerProvider(Protocol):
    def generate(self, generation_input: AnswerGenerationInput) -> str:
        pass


def generate_answer(review: EvidenceReview, provider: AnswerProvider) -> AnswerResult:
    answer_output = build_answer_output(review)

    if not review.is_sufficient:
        return answer_output

    try:
        answer_text = provider.generate(
            AnswerGenerationInput(
                question=answer_output.question,
                contexts=answer_output.contexts,
            )
        )
    except Exception:
        return build_generation_fallback_output(review, PROVIDER_ERROR_FALLBACK, reason="provider_error")

    answer_text = answer_text.strip()
    if not answer_text:
        return build_generation_fallback_output(review, EMPTY_ANSWER_FALLBACK, reason="empty_answer")

    return build_answer_output(review, answer_text=answer_text)