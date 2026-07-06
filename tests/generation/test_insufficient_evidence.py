from src.app_services.retrieval_service import RetrievalResult
from src.generation.citations import Citation
from src.generation.evidence import EvidenceStatus, evaluate_evidence


def make_result(
    chunk_id: str,
    fused_score: float,
    source_file: str = "policy-a.pdf",
    page_number: int = 2,
    content: str = "客户经理被投诉一次会影响评聘。",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        page_number=page_number,
        metadata={"chunk_index": 0},
        vector_score=fused_score,
        bm25_score=0.0,
        fused_score=fused_score,
    )


def test_evaluate_evidence_returns_insufficient_when_no_results():
    review = evaluate_evidence("客户经理投诉会影响评聘吗？", [])

    assert review.status == EvidenceStatus.INSUFFICIENT
    assert review.is_sufficient is False
    assert review.question == "客户经理投诉会影响评聘吗？"
    assert review.selected_results == []
    assert review.citations == []
    assert review.retrieval_summary == {
        "retrieved_count": 0,
        "selected_count": 0,
        "max_score": None,
        "min_score": 0.2,
        "reason": "no_retrieval_results",
    }


def test_evaluate_evidence_returns_insufficient_when_top_score_below_threshold():
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1", fused_score=0.19)],
        min_score=0.2,
    )

    assert review.status == EvidenceStatus.INSUFFICIENT
    assert review.is_sufficient is False
    assert review.selected_results == []
    assert review.citations == []
    assert review.retrieval_summary["retrieved_count"] == 1
    assert review.retrieval_summary["selected_count"] == 0
    assert review.retrieval_summary["max_score"] == 0.19
    assert review.retrieval_summary["reason"] == "score_below_threshold"


def test_evaluate_evidence_returns_selected_results_and_citations_when_sufficient():
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [make_result("chunk-1", fused_score=0.6)],
        min_score=0.2,
    )

    assert review.status == EvidenceStatus.SUFFICIENT
    assert review.is_sufficient is True
    assert [result.chunk_id for result in review.selected_results] == ["chunk-1"]
    assert review.citations == [
        Citation(
            source_file="policy-a.pdf",
            page_number=2,
            chunk_ids=("chunk-1",),
            metadata={"chunk_indexes": (0,)},
        )
    ]
    assert review.retrieval_summary["reason"] == "sufficient_evidence"


def test_evaluate_evidence_limits_selected_results_by_top_k():
    review = evaluate_evidence(
        "客户经理投诉会影响评聘吗？",
        [
            make_result("chunk-1", fused_score=0.7),
            make_result("chunk-2", fused_score=0.6),
            make_result("chunk-3", fused_score=0.5),
        ],
        top_k=2,
    )

    assert [result.chunk_id for result in review.selected_results] == ["chunk-1", "chunk-2"]
    assert review.retrieval_summary["retrieved_count"] == 3
    assert review.retrieval_summary["selected_count"] == 2