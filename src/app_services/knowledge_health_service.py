from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


# ── Default test queries for bank policy domain ───────────────────────────────

DEFAULT_QUERIES: list[str] = [
    "客户经理被投诉一次会影响评聘吗？",
    "月度考核和季度考核的权重分别是多少？",
    "投诉处理流程是什么？",
    "客户经理评聘申报时间是什么时候？",
    "考核扣分标准有哪些情形？",
]


# ── Status & Result ───────────────────────────────────────────────────────────

class KnowledgeHealthStatus(StrEnum):
    READY = "ready"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True)
class KnowledgeHealthResult:
    status: KnowledgeHealthStatus
    coverage_score: float
    missing_knowledge: list[dict]
    completeness_analysis: str
    checked_queries: int
    message: str


# ── LCEL chain ────────────────────────────────────────────────────────────────

_HEALTH_PROMPT = PromptTemplate.from_template(
    "你是知识库完整性检查专家。请分析以下检索结果，评估知识库对这些问题的覆盖情况。\n\n"
    "检索结果：\n{queries_with_results}\n\n"
    "请严格按以下 JSON 格式返回，不要包含任何额外文字：\n"
    "{{\n"
    '  "coverage_score": <0到1之间的小数>,\n'
    '  "missing_knowledge": [\n'
    "    {{\n"
    '      "query": "<测试问题>",\n'
    '      "missing_aspect": "<缺失的知识方面>",\n'
    '      "importance": "<高|中|低>",\n'
    '      "suggested_content": "<建议补充的内容>"\n'
    "    }}\n"
    "  ],\n"
    '  "completeness_analysis": "<整体完整性分析>"\n'
    "}}"
)


def build_health_check_chain(llm: Any):
    """LCEL chain: {queries_with_results} → parsed dict with coverage report."""
    return _HEALTH_PROMPT | llm | JsonOutputParser()


# ── Service function ──────────────────────────────────────────────────────────

def check_knowledge_health(
    queries: list[str] | None,
    retriever: Any,
    llm: Any,
) -> KnowledgeHealthResult:
    test_queries = queries if queries is not None else DEFAULT_QUERIES

    if not test_queries:
        return KnowledgeHealthResult(
            status=KnowledgeHealthStatus.EMPTY,
            coverage_score=0.0,
            missing_knowledge=[],
            completeness_analysis="",
            checked_queries=0,
            message="未提供测试问题，无法执行健康检查。",
        )

    # Retrieve docs for each query and build context
    query_results: list[str] = []
    for query in test_queries:
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = []

        if docs:
            snippets = "\n".join(
                f"  - {d.page_content[:150]}" for d in docs[:3]
            )
            query_results.append(f"问题：{query}\n检索到的内容：\n{snippets}")
        else:
            query_results.append(f"问题：{query}\n检索到的内容：无")

    queries_with_results = "\n\n".join(query_results)

    try:
        chain = build_health_check_chain(llm)
        parsed = chain.invoke({"queries_with_results": queries_with_results})

        return KnowledgeHealthResult(
            status=KnowledgeHealthStatus.READY,
            coverage_score=float(parsed.get("coverage_score", 0.0)),
            missing_knowledge=parsed.get("missing_knowledge", []),
            completeness_analysis=parsed.get("completeness_analysis", ""),
            checked_queries=len(test_queries),
            message=f"健康检查完成，覆盖率评分：{parsed.get('coverage_score', 0.0):.0%}",
        )
    except Exception as exc:
        return KnowledgeHealthResult(
            status=KnowledgeHealthStatus.FAILED,
            coverage_score=0.0,
            missing_knowledge=[],
            completeness_analysis="",
            checked_queries=len(test_queries),
            message=f"健康检查失败：{type(exc).__name__}",
        )
