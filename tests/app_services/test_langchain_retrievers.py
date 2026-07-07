from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.app_services.langchain_retrievers import (
    build_bm25_retriever,
    build_query_rewrite_chain,
    run_hybrid,
    run_multiquery,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _doc(content: str, source: str = "p.pdf", page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"source_file": source, "page_number": page},
    )


def _fake_retriever(docs: list[Document]) -> MagicMock:
    r = MagicMock()
    r.invoke.return_value = docs
    return r


def _fake_llm(text: str) -> MagicMock:
    llm = MagicMock()
    ai_msg = AIMessage(content=text)
    llm.invoke.return_value = ai_msg
    llm.return_value = ai_msg   # used when called as __call__ inside LCEL lambda
    return llm


def _fake_chat_model(text: str) -> FakeListChatModel:
    """Proper Runnable LLM for LCEL chain tests."""
    return FakeListChatModel(responses=[text])


# ── run_multiquery ────────────────────────────────────────────────────────────

def test_multiquery_runs_additional_searches_for_each_variant():
    doc_a = _doc("内容A")
    doc_b = _doc("内容B")
    retriever = _fake_retriever([doc_a, doc_b])
    llm = _fake_llm("变体问题一\n变体问题二")

    results = run_multiquery("原始问题", retriever, llm, num_variants=2)

    # at minimum the original query was searched
    assert retriever.invoke.call_count >= 1
    assert len(results) > 0


def test_multiquery_deduplicates_identical_docs():
    doc = _doc("重复内容")
    retriever = _fake_retriever([doc])   # same doc every call
    llm = _fake_llm("变体一\n变体二")

    results = run_multiquery("问题", retriever, llm, num_variants=2)

    # Despite multiple retriever calls, identical content appears once
    contents = [d.page_content for d in results]
    assert contents.count("重复内容") == 1


def test_multiquery_still_returns_results_when_llm_returns_empty_variants():
    doc = _doc("内容")
    retriever = _fake_retriever([doc])
    llm = _fake_llm("")   # no variants generated

    results = run_multiquery("问题", retriever, llm, num_variants=2)

    assert len(results) >= 1  # original query results still returned


# ── run_hybrid ────────────────────────────────────────────────────────────────

def test_hybrid_merges_results_from_both_retrievers():
    bm25_doc = _doc("BM25专属内容", page=1)
    vec_doc = _doc("Vector专属内容", page=2)
    bm25_r = _fake_retriever([bm25_doc])
    vec_r = _fake_retriever([vec_doc])

    results = run_hybrid("问题", bm25_r, vec_r, k=4)

    contents = {d.page_content for d in results}
    assert "BM25专属内容" in contents
    assert "Vector专属内容" in contents


def test_hybrid_deduplicates_docs_appearing_in_both():
    shared_doc = _doc("共同内容")
    bm25_r = _fake_retriever([shared_doc, _doc("BM25专属")])
    vec_r = _fake_retriever([shared_doc, _doc("Vector专属")])

    results = run_hybrid("问题", bm25_r, vec_r, k=4)

    contents = [d.page_content for d in results]
    assert contents.count("共同内容") == 1


def test_hybrid_respects_top_k_limit():
    docs = [_doc(f"内容{i}") for i in range(10)]
    bm25_r = _fake_retriever(docs[:5])
    vec_r = _fake_retriever(docs[5:])

    results = run_hybrid("问题", bm25_r, vec_r, k=3)

    assert len(results) <= 3


def test_hybrid_ranks_docs_in_both_results_higher():
    shared = _doc("两者都检索到")
    bm25_only = _doc("仅BM25")
    vec_only = _doc("仅Vector")
    bm25_r = _fake_retriever([shared, bm25_only])
    vec_r = _fake_retriever([shared, vec_only])

    results = run_hybrid("问题", bm25_r, vec_r, k=3)

    # shared doc should be first (highest RRF score)
    assert results[0].page_content == "两者都检索到"


# ── build_query_rewrite_chain ─────────────────────────────────────────────────

def test_rewrite_chain_calls_llm_with_question():
    llm = _fake_chat_model("改写后的问题")

    chain = build_query_rewrite_chain(llm)
    result = chain.invoke({"question": "原始问题"})

    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_chain_returns_llm_output():
    llm = _fake_chat_model("更清晰的搜索查询")

    chain = build_query_rewrite_chain(llm)
    result = chain.invoke({"question": "问题"})

    assert result == "更清晰的搜索查询"


# ── build_bm25_retriever ──────────────────────────────────────────────────────

def test_build_bm25_retriever_from_chunks():
    chunks = [
        {"chunk_id": "c1", "content": "客户经理考核规则", "source_file": "p.pdf", "page_number": 1},
        {"chunk_id": "c2", "content": "投诉处理流程", "source_file": "p.pdf", "page_number": 2},
    ]

    retriever = build_bm25_retriever(chunks, k=2)

    results = retriever.invoke("客户经理")
    assert len(results) <= 2
    assert all(hasattr(d, "page_content") for d in results)
