from dataclasses import dataclass

from src.app_services.knowledge_base_loader import KnowledgeBaseLoadResult, KnowledgeBaseStatus
from src.app_services.retrieval_service import RetrievalMode, RetrievalResult
from src.app_services.ui_answer_service import UiAnswerService
from src.generation.answer_contract import AnswerStatus


class FakeRetrievalService:
    def search(self, query, mode=RetrievalMode.HYBRID, top_k=4, alpha=0.5):
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                content="客户经理被投诉一次会影响评聘。",
                source_file="policy-a.pdf",
                page_number=2,
                metadata={"chunk_index": 0},
                vector_score=0.0,
                bm25_score=0.8,
                fused_score=0.8,
            )
        ][:top_k]


@dataclass(frozen=True)
class FakeAnswerProvider:
    def generate(self, generation_input):
        return "会影响，需结合投诉性质和处理结果判断。"


def test_ui_answer_service_answers_with_loaded_knowledge_base():
    service = UiAnswerService(
        knowledge_base=KnowledgeBaseLoadResult(
            status=KnowledgeBaseStatus.READY,
            chunk_count=1,
            retrieval_service=FakeRetrievalService(),
            message="已加载 1 个知识库 chunk。",
        ),
        answer_provider=FakeAnswerProvider(),
    )

    answer = service.answer("客户经理投诉会影响评聘吗？")

    assert answer.status == AnswerStatus.ANSWER_READY
    assert answer.answer_text == "会影响，需结合投诉性质和处理结果判断。"
    assert answer.citations[0].source_file == "policy-a.pdf"


def test_ui_answer_service_returns_fallback_when_knowledge_base_is_missing():
    service = UiAnswerService(
        knowledge_base=KnowledgeBaseLoadResult(
            status=KnowledgeBaseStatus.MISSING,
            chunk_count=0,
            retrieval_service=None,
            message="未找到知识库 chunks.jsonl，请先构建或加载知识库。",
        ),
        answer_provider=FakeAnswerProvider(),
    )

    answer = service.answer("客户经理投诉会影响评聘吗？")

    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.fallback_message == "未找到知识库 chunks.jsonl，请先构建或加载知识库。"
    assert answer.retrieval_summary == {
        "retrieved_count": 0,
        "selected_count": 0,
        "max_score": None,
        "min_score": 0.2,
        "reason": "knowledge_base_missing",
    }