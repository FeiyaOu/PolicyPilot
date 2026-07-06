from src.app_services.retrieval_service import RetrievalResult
from src.generation.citations import Citation, build_citations


def make_result(
    chunk_id: str,
    source_file: str,
    page_number: int,
    chunk_index: int,
    content: str = "客户经理被投诉一次会影响评聘。",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        page_number=page_number,
        metadata={"chunk_index": chunk_index},
        vector_score=0.8,
        bm25_score=0.4,
        fused_score=0.6,
    )


def test_build_citations_from_retrieval_results_source_metadata():
    citations = build_citations([
        make_result("chunk-1", "policy-a.pdf", 2, 0),
        make_result("chunk-2", "policy-b.pdf", 5, 1),
    ])

    assert citations == [
        Citation(
            source_file="policy-a.pdf",
            page_number=2,
            chunk_ids=("chunk-1",),
            metadata={"chunk_indexes": (0,)},
        ),
        Citation(
            source_file="policy-b.pdf",
            page_number=5,
            chunk_ids=("chunk-2",),
            metadata={"chunk_indexes": (1,)},
        ),
    ]


def test_build_citations_deduplicates_same_source_and_page():
    citations = build_citations([
        make_result("chunk-1", "policy-a.pdf", 2, 0),
        make_result("chunk-2", "policy-a.pdf", 2, 1),
    ])

    assert citations == [
        Citation(
            source_file="policy-a.pdf",
            page_number=2,
            chunk_ids=("chunk-1", "chunk-2"),
            metadata={"chunk_indexes": (0, 1)},
        )
    ]


def test_build_citations_uses_fallback_for_missing_source_fields():
    result = RetrievalResult(
        chunk_id="chunk-1",
        content="缺少来源字段的内容。",
        source_file="",
        page_number=0,
        metadata={},
        vector_score=0.0,
        bm25_score=0.0,
        fused_score=0.0,
    )

    citations = build_citations([result])

    assert citations == [
        Citation(
            source_file="unknown source",
            page_number=None,
            chunk_ids=("chunk-1",),
            metadata={"chunk_indexes": ()},
        )
    ]


def test_build_citations_returns_empty_list_for_no_retrieval_results():
    assert build_citations([]) == []
