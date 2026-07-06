from dataclasses import dataclass

from src.app_services.demo_answer_service import DemoAnswerProvider, DemoAnswerService
from src.app_services.retrieval_service import RetrievalMode, RetrievalResult
from src.generation.answer_contract import AnswerStatus


class FakeRetrievalService:
    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.received_query = None
        self.received_mode = None
        self.received_top_k = None

    def search(self, query, mode=RetrievalMode.HYBRID, top_k=4, alpha=0.5):
        self.received_query = query
        self.received_mode = mode
        self.received_top_k = top_k
        return self.results[:top_k]


@dataclass(frozen=True)
class FakeAnswerProvider:
    answer_text: str = "会影响，系统会依据投诉性质、处理结果和制度条款综合判断。"

    def generate(self, generation_input):
        return self.answer_text


def make_result(
    chunk_id: str,
    content: str = "客户经理被投诉一次会影响评聘。",
    fused_score: float = 0.6,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        source_file="policy-a.pdf",
        page_number=2,
        metadata={"chunk_index": 0},
        vector_score=fused_score,
        bm25_score=0.0,
        fused_score=fused_score,
    )


def test_demo_answer_service_returns_answer_ready_result_with_citations():
    retrieval_service = FakeRetrievalService([make_result("chunk-1")])
    service = DemoAnswerService(
        retrieval_service=retrieval_service,
        answer_provider=FakeAnswerProvider(),
    )

    answer = service.answer("客户经理投诉会影响评聘吗？", top_k=2)

    assert retrieval_service.received_query == "客户经理投诉会影响评聘吗？"
    assert retrieval_service.received_mode == RetrievalMode.HYBRID
    assert retrieval_service.received_top_k == 2
    assert answer.status == AnswerStatus.ANSWER_READY
    assert answer.answer_text == "会影响，系统会依据投诉性质、处理结果和制度条款综合判断。"
    assert answer.contexts[0]["chunk_id"] == "chunk-1"
    assert answer.citations[0].source_file == "policy-a.pdf"
    assert answer.citations[0].page_number == 2


def test_demo_answer_service_returns_fallback_when_retrieval_is_empty():
    service = DemoAnswerService(
        retrieval_service=FakeRetrievalService([]),
        answer_provider=FakeAnswerProvider(),
    )

    answer = service.answer("客户经理投诉会影响评聘吗？")

    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.answer_text is None
    assert answer.fallback_message == "未检索到足够可靠的政策依据，暂时无法回答该问题。"
    assert answer.contexts == []
    assert answer.citations == []


def test_demo_answer_provider_summarizes_context_without_external_llm():
    provider = DemoAnswerProvider()

    answer_text = provider.generate(
        generation_input=type(
            "Input",
            (),
            {
                "contexts": [
                    {"content": "客户经理被投诉一次会影响评聘。"},
                    {"content": "投诉记录需要保留处理材料。"},
                ]
            },
        )()
    )

    assert answer_text == "根据已检索到的政策依据：客户经理被投诉一次会影响评聘。 投诉记录需要保留处理材料。"