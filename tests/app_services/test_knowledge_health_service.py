from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.app_services.knowledge_health_service import (
    KnowledgeHealthResult,
    KnowledgeHealthStatus,
    build_health_check_chain,
    check_knowledge_health,
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


_VALID_JSON = json.dumps({
    "coverage_score": 0.75,
    "missing_knowledge": [
        {
            "query": "季度考核权重",
            "missing_aspect": "季度考核细则",
            "importance": "高",
            "suggested_content": "建议补充季度考核章节",
        }
    ],
    "completeness_analysis": "覆盖投诉处理，缺少季度考核细则",
})


# ── build_health_check_chain ──────────────────────────────────────────────────

def test_build_health_check_chain_returns_callable_runnable():
    llm = FakeListChatModel(responses=[_VALID_JSON])
    chain = build_health_check_chain(llm)

    result = chain.invoke({"queries_with_results": "查询内容"})

    assert isinstance(result, dict)
    assert "coverage_score" in result


def test_health_check_chain_parses_json_output():
    llm = FakeListChatModel(responses=[_VALID_JSON])
    chain = build_health_check_chain(llm)

    result = chain.invoke({"queries_with_results": "任意内容"})

    assert result["coverage_score"] == 0.75
    assert len(result["missing_knowledge"]) == 1
    assert result["missing_knowledge"][0]["importance"] == "高"


# ── check_knowledge_health ────────────────────────────────────────────────────

def test_check_health_returns_ready_result_with_parsed_fields():
    retriever = _retriever([_doc("客户经理考核办法内容")])
    llm = FakeListChatModel(responses=[_VALID_JSON])

    result = check_knowledge_health(
        queries=["客户经理考核标准是什么？"],
        retriever=retriever,
        llm=llm,
    )

    assert isinstance(result, KnowledgeHealthResult)
    assert result.status == KnowledgeHealthStatus.READY
    assert result.coverage_score == 0.75
    assert result.checked_queries == 1
    assert len(result.missing_knowledge) == 1


def test_check_health_calls_retriever_for_each_query():
    retriever = _retriever([_doc("内容")])
    llm = FakeListChatModel(responses=[_VALID_JSON])

    check_knowledge_health(
        queries=["问题一", "问题二", "问题三"],
        retriever=retriever,
        llm=llm,
    )

    assert retriever.invoke.call_count == 3


def test_check_health_returns_failed_when_llm_returns_invalid_json():
    retriever = _retriever([_doc("内容")])
    llm = FakeListChatModel(responses=["这不是JSON格式"])

    result = check_knowledge_health(
        queries=["问题"],
        retriever=retriever,
        llm=llm,
    )

    assert result.status == KnowledgeHealthStatus.FAILED


def test_check_health_handles_empty_query_list():
    retriever = _retriever([])
    llm = FakeListChatModel(responses=[_VALID_JSON])

    result = check_knowledge_health(
        queries=[],
        retriever=retriever,
        llm=llm,
    )

    assert result.status == KnowledgeHealthStatus.EMPTY
    assert result.checked_queries == 0
    assert not retriever.invoke.called


def test_check_health_includes_completeness_analysis():
    retriever = _retriever([_doc("内容")])
    llm = FakeListChatModel(responses=[_VALID_JSON])

    result = check_knowledge_health(
        queries=["问题"],
        retriever=retriever,
        llm=llm,
    )

    assert result.completeness_analysis == "覆盖投诉处理，缺少季度考核细则"


def test_check_health_uses_default_queries_when_none_provided():
    """Service has built-in default test queries for bank policy domain."""
    retriever = _retriever([_doc("考核内容")])
    llm = FakeListChatModel(responses=[_VALID_JSON])

    result = check_knowledge_health(
        queries=None,   # use defaults
        retriever=retriever,
        llm=llm,
    )

    assert result.status == KnowledgeHealthStatus.READY
    assert result.checked_queries > 0
