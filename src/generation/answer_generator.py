from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.generation.answer_contract import AnswerResult, build_answer_output
from src.generation.evidence import EvidenceReview


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

    answer_text = provider.generate(
        AnswerGenerationInput(
            question=answer_output.question,
            contexts=answer_output.contexts,
        )
    )

    return build_answer_output(review, answer_text=answer_text)