from src.app_services.retrieval_service import RetrievalResult
from src.generation.answer_contract import AnswerStatus, build_answer_output
from src.generation.citations import Citation
from src.generation.evidence import evaluate_evidence


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


def test_build_answer_output_returns_answer_ready_contract_for_sufficient_evidence():
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = build_answer_output(review)

    assert answer.question == "客户经理投诉会影响评聘吗？"
    assert answer.status == AnswerStatus.ANSWER_READY
    assert answer.answer_text is None
    assert answer.fallback_message is None
    assert answer.contexts == [
        {
            "chunk_id": "chunk-1",
            "content": "客户经理被投诉一次会影响评聘。",
            "source_file": "policy-a.pdf",
            "page_number": 2,
            "score": 0.6,
        }
    ]
    assert answer.citations == [
        Citation(
            source_file="policy-a.pdf",
            page_number=2,
            chunk_ids=("chunk-1",),
            metadata={"chunk_indexes": (0,)},
        )
    ]
    assert answer.retrieval_summary["reason"] == "sufficient_evidence"


def test_build_answer_output_returns_fallback_contract_for_insufficient_evidence():
    review = evaluate_evidence("客户经理投诉会影响评聘吗？", [])

    answer = build_answer_output(review)

    assert answer.question == "客户经理投诉会影响评聘吗？"
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.answer_text is None
    assert answer.fallback_message == "未检索到足够可靠的政策依据，暂时无法回答该问题。"
    assert answer.contexts == []
    assert answer.citations == []
    assert answer.retrieval_summary == review.retrieval_summary


def test_build_answer_output_accepts_generated_answer_text_when_available():
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1")],
    )

    answer = build_answer_output(review, answer_text="会影响，需结合投诉性质和处理结果判断。")

    assert answer.status == AnswerStatus.ANSWER_READY
    assert answer.answer_text == "会影响，需结合投诉性质和处理结果判断。"
    assert answer.fallback_message is None