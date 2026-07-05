import pytest

from src.retrieval.hybrid import HybridSearchResult, fuse_hybrid_scores


def test_fuse_hybrid_scores_averages_vector_and_bm25_scores_when_alpha_is_half():
    results = fuse_hybrid_scores(
        vector_scores={"chunk-a": 0.8},
        bm25_scores={"chunk-a": 0.4},
        alpha=0.5,
    )

    assert results[0].chunk_id == "chunk-a"
    assert results[0].vector_score == 0.8
    assert results[0].bm25_score == 0.4
    assert results[0].fused_score == pytest.approx(0.6)


def test_fuse_hybrid_scores_respects_alpha_edges():
    vector_only = fuse_hybrid_scores(
        vector_scores={"chunk-a": 0.8},
        bm25_scores={"chunk-a": 0.4},
        alpha=1.0,
    )
    bm25_only = fuse_hybrid_scores(
        vector_scores={"chunk-a": 0.8},
        bm25_scores={"chunk-a": 0.4},
        alpha=0.0,
    )

    assert vector_only[0].fused_score == 0.8
    assert bm25_only[0].fused_score == 0.4


def test_fuse_hybrid_scores_sorts_by_fused_score_descending_and_uses_zero_for_missing_scores():
    results = fuse_hybrid_scores(
        vector_scores={"chunk-a": 0.9, "chunk-b": 0.2},
        bm25_scores={"chunk-b": 1.0, "chunk-c": 0.7},
        alpha=0.5,
    )

    assert [result.chunk_id for result in results] == ["chunk-b", "chunk-a", "chunk-c"]
    assert [result.fused_score for result in results] == [pytest.approx(0.6), pytest.approx(0.45), pytest.approx(0.35)]


def test_fuse_hybrid_scores_rejects_alpha_outside_zero_to_one():
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        fuse_hybrid_scores(vector_scores={}, bm25_scores={}, alpha=1.1)
