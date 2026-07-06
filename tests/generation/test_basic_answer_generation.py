from dataclasses import dataclass

from src.app_services.retrieval_service import RetrievalResult
from src.generation.answer_contract import AnswerStatus
from src.generation.answer_generator import AnswerGenerationInput, generate_answer
from src.generation.evidence import evaluate_evidence


@dataclass
class FakeAnswerProvider:
    answer_text: str = "会影响，需结合投诉性质、处理结果和制度条款判断。"
    received_input: AnswerGenerationInput | None = None

    def generate(self, generation_input: AnswerGenerationInput) -> str:
        self.received_input = generation_input
        return self.answer_text


@dataclass
class FailingAnswerProvider:
    received_input: AnswerGenerationInput | None = None

    def generate(self, generation_input: AnswerGenerationInput) -> str:
        self.received_input = generation_input
        raise RuntimeError("secret-key should not be exposed")


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


def test_generate_answer_returns_fallback_without_calling_provider_when_evidence_insufficient():
    provider = FakeAnswerProvider()
    review = evaluate_evidence("客户经理投诉会影响评聘吗？", [])

    answer = generate_answer(review, provider)

    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.answer_text is None
    assert answer.fallback_message == "未检索到足够可靠的政策依据，暂时无法回答该问题。"
    assert provider.received_input is None


def test_generate_answer_calls_provider_with_selected_context_when_evidence_is_sufficient():
    provider = FakeAnswerProvider()
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = generate_answer(review, provider)

    assert provider.received_input == AnswerGenerationInput(
        question="客户经理投诉会影响评聘吗？",
        contexts=[
            {
                "chunk_id": "chunk-1",
                "content": "客户经理被投诉一次会影响评聘。",
                "source_file": "policy-a.pdf",
                "page_number": 2,
                "score": 0.6,
            }
        ],
    )
    assert answer.status == AnswerStatus.ANSWER_READY
    assert answer.answer_text == "会影响，需结合投诉性质、处理结果和制度条款判断。"
    assert answer.fallback_message is None


def test_generate_answer_preserves_citations_and_retrieval_summary():
    provider = FakeAnswerProvider()
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = generate_answer(review, provider)

    assert answer.citations == review.citations
    assert answer.retrieval_summary == review.retrieval_summary


def test_generate_answer_returns_safe_fallback_when_provider_raises():
    provider = FailingAnswerProvider()
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = generate_answer(review, provider)

    assert answer.status == AnswerStatus.GENERATION_FAILED
    assert answer.answer_text is None
    assert answer.fallback_message == "模型回答生成失败，请稍后重试或切换本地演示回答。"
    assert "secret-key" not in answer.fallback_message
    assert answer.contexts[0]["chunk_id"] == "chunk-1"
    assert answer.citations == review.citations
    assert answer.retrieval_summary["reason"] == "sufficient_evidence"
    assert answer.retrieval_summary["generation_status"] == "failed"
    assert answer.retrieval_summary["generation_reason"] == "provider_error"


def test_generate_answer_rejects_blank_provider_response():
    provider = FakeAnswerProvider(answer_text="  \n\t")
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = generate_answer(review, provider)

    assert answer.status == AnswerStatus.GENERATION_FAILED
    assert answer.answer_text is None
    assert answer.fallback_message == "模型返回为空，已拒绝生成无依据答案。"
    assert answer.contexts[0]["chunk_id"] == "chunk-1"
    assert answer.retrieval_summary["generation_reason"] == "empty_answer"


def test_generate_answer_strips_provider_response_before_returning():
    provider = FakeAnswerProvider(answer_text="  会影响，需结合投诉性质和处理结果判断。\n")
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = generate_answer(review, provider)

    assert answer.status == AnswerStatus.ANSWER_READY
    assert answer.answer_text == "会影响，需结合投诉性质和处理结果判断。"