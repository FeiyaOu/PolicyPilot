from src.retrieval.multi_query import RetrievalHit, merge_multi_query_results, normalize_query_variants


def test_normalize_query_variants_includes_original_query_and_removes_empty_duplicates():
    variants = normalize_query_variants(
        original_query="投诉会影响客户经理评聘吗？",
        query_variants=["", "客户经理投诉记录如何影响评聘", "投诉会影响客户经理评聘吗？"],
    )

    assert variants == (
        "投诉会影响客户经理评聘吗？",
        "客户经理投诉记录如何影响评聘",
    )


def test_normalize_query_variants_falls_back_to_original_query_when_variants_are_empty():
    variants = normalize_query_variants(
        original_query="客户经理每年评聘申报时间是什么时候？",
        query_variants=[],
    )

    assert variants == ("客户经理每年评聘申报时间是什么时候？",)


def test_merge_multi_query_results_deduplicates_by_chunk_id_and_keeps_highest_score():
    result = merge_multi_query_results(
        original_query="投诉会影响客户经理评聘吗？",
        query_variants=["客户经理投诉记录如何影响评聘", "投诉扣分是否影响年度评聘"],
        hits_by_query={
            "投诉会影响客户经理评聘吗？": [
                RetrievalHit(chunk_id="chunk-1", score=0.72, content="投诉记录会影响评聘。"),
                RetrievalHit(chunk_id="chunk-2", score=0.64, content="评聘材料需按时提交。"),
            ],
            "客户经理投诉记录如何影响评聘": [
                RetrievalHit(chunk_id="chunk-1", score=0.91, content="投诉记录会影响评聘。"),
            ],
        },
    )

    assert result.original_query == "投诉会影响客户经理评聘吗？"
    assert result.query_variants == (
        "投诉会影响客户经理评聘吗？",
        "客户经理投诉记录如何影响评聘",
        "投诉扣分是否影响年度评聘",
    )
    assert [(hit.chunk_id, hit.score) for hit in result.hits] == [("chunk-1", 0.91), ("chunk-2", 0.64)]


def test_merge_multi_query_results_does_not_deduplicate_by_content_text():
    result = merge_multi_query_results(
        original_query="投诉会影响客户经理评聘吗？",
        query_variants=[],
        hits_by_query={
            "投诉会影响客户经理评聘吗？": [
                RetrievalHit(chunk_id="chunk-1", score=0.9, content="同一段政策文本"),
                RetrievalHit(chunk_id="chunk-2", score=0.8, content="同一段政策文本"),
            ],
        },
    )

    assert [hit.chunk_id for hit in result.hits] == ["chunk-1", "chunk-2"]
