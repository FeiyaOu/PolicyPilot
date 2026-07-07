from __future__ import annotations

import operator
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.app_services.adaptive_rag_graph import (
    AdaptiveRagRunResult,
    AdaptiveRagState,
    build_adaptive_rag_graph,
    run_adaptive_rag,
    score_docs,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _doc(content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"source_file": "policy.pdf", "page_number": 1},
    )


def _retriever(docs: list[Document]) -> MagicMock:
    r = MagicMock()
    r.invoke.return_value = docs
    return r


def _llm(text: str = "生成的答案") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=text)
    return llm


def _build_graph(bm25_docs, hybrid_docs=None, llm=None):
    bm25_r = _retriever(bm25_docs)
    vec_r = _retriever(hybrid_docs if hybrid_docs is not None else bm25_docs)
    return build_adaptive_rag_graph(
        bm25_retriever=bm25_r,
        vector_retriever=vec_r,
        llm=llm or _llm(),
    ), bm25_r, vec_r


# ── score_docs ────────────────────────────────────────────────────────────────

def test_score_docs_returns_zero_for_empty_docs():
    assert score_docs([], "任何问题") == 0.0


def test_score_docs_returns_positive_when_content_matches_question():
    docs = [_doc("客户经理投诉影响评聘")]
    score = score_docs(docs, "客户经理被投诉")
    assert score > 0.0


def test_score_docs_returns_lower_score_when_content_unrelated():
    docs = [_doc("天气晴朗风和日丽")]
    score = score_docs(docs, "客户经理被投诉影响评聘")
    assert score < 1.0


# ── graph: direct BM25 path ──────────────────────────────────────────────────

def test_graph_generates_answer_directly_when_bm25_sufficient():
    # BM25 returns a doc whose content matches the question well
    bm25_docs = [_doc("客户经理被投诉一次会影响评聘，需根据核查结果处理")]
    graph, _, _ = _build_graph(bm25_docs)

    result = run_adaptive_rag(graph, question="客户经理被投诉影响评聘")

    assert result.answer != ""
    assert "BM25" in result.retrieval_path[0]
    assert result.retrieval_mode in ("bm25", "hybrid")


def test_graph_answer_contains_llm_output_on_bm25_hit():
    bm25_docs = [_doc("客户经理投诉处理规则")]
    graph, _, _ = _build_graph(bm25_docs, llm=_llm("测试答案内容"))

    result = run_adaptive_rag(graph, question="客户经理被投诉")

    assert "测试答案内容" in result.answer


# ── graph: escalation to hybrid ──────────────────────────────────────────────

def test_graph_escalates_to_hybrid_when_bm25_returns_no_docs():
    hybrid_docs = [_doc("客户经理被投诉一次会影响评聘")]
    graph, bm25_r, vec_r = _build_graph(bm25_docs=[], hybrid_docs=hybrid_docs)

    result = run_adaptive_rag(graph, question="客户经理被投诉影响评聘")

    # BM25 was tried but returned nothing → escalated to hybrid
    assert bm25_r.invoke.called
    assert vec_r.invoke.called   # hybrid triggers vector retrieval
    assert result.retrieval_mode == "hybrid"


def test_graph_retrieval_path_records_escalation():
    hybrid_docs = [_doc("客户经理被投诉一次会影响评聘")]
    graph, _, _ = _build_graph(bm25_docs=[], hybrid_docs=hybrid_docs)

    result = run_adaptive_rag(graph, question="客户经理被投诉")

    path_str = " ".join(result.retrieval_path)
    assert "BM25" in path_str
    assert len(result.retrieval_path) > 1  # path has more than just BM25


# ── graph: fallback ───────────────────────────────────────────────────────────

def test_graph_returns_fallback_when_both_retrievers_return_no_docs():
    graph, _, _ = _build_graph(bm25_docs=[], hybrid_docs=[])

    result = run_adaptive_rag(graph, question="无关问题xyz")

    assert result.retrieval_mode == "fallback"
    assert result.answer != ""
    assert result.sources == []


def test_graph_fallback_path_includes_fallback_label():
    graph, _, _ = _build_graph(bm25_docs=[], hybrid_docs=[])

    result = run_adaptive_rag(graph, question="无关问题xyz")

    path_str = " ".join(result.retrieval_path)
    assert "fallback" in path_str.lower() or "Fallback" in path_str


# ── result structure ──────────────────────────────────────────────────────────

def test_run_adaptive_rag_returns_typed_result():
    bm25_docs = [_doc("内容")]
    graph, _, _ = _build_graph(bm25_docs)

    result = run_adaptive_rag(graph, question="问题")

    assert isinstance(result, AdaptiveRagRunResult)
    assert isinstance(result.answer, str)
    assert isinstance(result.sources, list)
    assert isinstance(result.retrieval_path, list)
    assert isinstance(result.evidence_score, float)


def test_run_adaptive_rag_passes_chat_history_to_generate():
    bm25_docs = [_doc("客户经理投诉规则处理")]
    llm = _llm("答案")
    graph, _, _ = _build_graph(bm25_docs, llm=llm)

    run_adaptive_rag(
        graph,
        question="客户经理投诉",   # matches doc content → BM25 sufficient → LLM called
        chat_history=[
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
        ],
    )

    messages = llm.invoke.call_args[0][0]
    all_content = " ".join(m.content for m in messages if hasattr(m, "content"))
    assert "第一轮问题" in all_content
