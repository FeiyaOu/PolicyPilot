from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.app_services.retrieval_service import RetrievalMode, RetrievalResult
from src.generation.answer_contract import AnswerResult
from src.generation.answer_generator import AnswerGenerationInput, AnswerProvider, generate_answer
from src.generation.evidence import evaluate_evidence


class RetrievalServiceLike(Protocol):
    def search(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 4,
        alpha: float = 0.5,
    ) -> list[RetrievalResult]:
        pass


@dataclass(frozen=True)
class DemoAnswerProvider(AnswerProvider):
    def generate(self, generation_input: AnswerGenerationInput) -> str:
        evidence_text = " ".join(context["content"] for context in generation_input.contexts)
        return f"根据已检索到的政策依据：{evidence_text}"


@dataclass(frozen=True)
class DemoAnswerService:
    retrieval_service: RetrievalServiceLike
    answer_provider: AnswerProvider

    def answer(
        self,
        question: str,
        top_k: int = 4,
        min_score: float = 0.2,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> AnswerResult:
        retrieval_results = self.retrieval_service.search(question, mode=mode, top_k=top_k)
        review = evaluate_evidence(
            question=question,
            retrieval_results=retrieval_results,
            min_score=min_score,
            top_k=top_k,
        )
        return generate_answer(review, self.answer_provider)


class StaticDemoRetrievalService:
    def search(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 4,
        alpha: float = 0.5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []

        return [
            RetrievalResult(
                chunk_id="demo-policy-1-page-2-chunk-0",
                content="客户经理被投诉一次会影响评聘，需结合投诉性质、核查结果和整改情况综合判断。",
                source_file="客户经理绩效评聘办法.pdf",
                page_number=2,
                metadata={"chunk_index": 0},
                vector_score=0.78,
                bm25_score=0.66,
                fused_score=0.72,
            ),
            RetrievalResult(
                chunk_id="demo-policy-2-page-8-chunk-1",
                content="投诉记录和处理材料应归档保存，作为后续合规检查和人员管理的重要依据。",
                source_file="网点服务投诉处理细则.pdf",
                page_number=8,
                metadata={"chunk_index": 1},
                vector_score=0.65,
                bm25_score=0.58,
                fused_score=0.62,
            ),
        ][:top_k]


def build_demo_answer_service() -> DemoAnswerService:
    return DemoAnswerService(
        retrieval_service=StaticDemoRetrievalService(),
        answer_provider=DemoAnswerProvider(),
    )