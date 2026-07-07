from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.app_services.langchain_retrievers import run_hybrid


_SCORE_THRESHOLD = 0.3

_SYSTEM_PROMPT = (
    "你是银行内部制度问答助手，仅根据以下参考文档回答问题。"
    "回答时请引用来源文件名和页码。"
)


# ── State ─────────────────────────────────────────────────────────────────────

class AdaptiveRagState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_docs: list[Document]
    evidence_score: float
    retrieval_mode: str                              # "bm25" | "hybrid" | "fallback"
    answer: str
    sources: list[dict]
    retrieval_path: Annotated[list[str], operator.add]  # accumulates across nodes
    chat_history: list[dict]
    summary: str


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdaptiveRagRunResult:
    answer: str
    sources: list[dict]
    retrieval_mode: str
    retrieval_path: list[str]
    evidence_score: float


# ── Pure helpers ──────────────────────────────────────────────────────────────

def score_docs(docs: list[Document], question: str) -> float:
    """Bigram character overlap between question and retrieved content."""
    if not docs:
        return 0.0

    bigrams = lambda s: {s[i: i + 2] for i in range(len(s) - 1)}
    q_bi = bigrams(question)
    if not q_bi:
        return 0.5

    all_content = " ".join(d.page_content for d in docs)
    c_bi = bigrams(all_content)
    return min(1.0, len(q_bi & c_bi) / len(q_bi))


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source_file", "未知来源")
        page = meta.get("page_number", "?")
        parts.append(f"[文档{i}] 来源：{source} 第{page}页\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[dict]:
    seen: set[tuple] = set()
    sources: list[dict] = []
    for doc in docs:
        meta = doc.metadata
        key = (meta.get("source_file"), meta.get("page_number"))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source_file": meta.get("source_file", ""),
                    "page_number": meta.get("page_number"),
                    "content": doc.page_content,
                }
            )
    return sources


def _build_messages(
    question: str,
    context: str,
    chat_history: list[dict],
    summary: str,
) -> list:
    msgs: list = [SystemMessage(content=_SYSTEM_PROMPT)]
    if summary:
        msgs.append(SystemMessage(content=f"历史对话摘要：\n{summary}"))
    for m in chat_history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=f"参考文档：\n{context}\n\n问题：{question}"))
    return msgs


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_adaptive_rag_graph(
    bm25_retriever: Any,
    vector_retriever: Any,
    llm: Any,
    score_threshold: float = _SCORE_THRESHOLD,
):
    """Build and compile the Adaptive RAG StateGraph.

    Graph structure:
        rewrite → retrieve_bm25 → evaluate_bm25
                                        ├── sufficient → generate → END
                                        └── insufficient → retrieve_hybrid
                                                              → evaluate_hybrid
                                                                    ├── sufficient → generate → END
                                                                    └── insufficient → fallback → END
    """

    # ── Node functions ────────────────────────────────────────────────────────

    def rewrite_node(state: AdaptiveRagState) -> dict:
        # Passthrough — query rewriting can be added here later
        return {"rewritten_query": state["question"]}

    def retrieve_bm25_node(state: AdaptiveRagState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        try:
            docs = bm25_retriever.invoke(query)
        except Exception:
            docs = []
        return {
            "retrieved_docs": docs,
            "retrieval_mode": "bm25",
            "retrieval_path": ["BM25"],
        }

    def evaluate_bm25_node(state: AdaptiveRagState) -> dict:
        score = score_docs(state["retrieved_docs"], state["question"])
        return {"evidence_score": score}

    def route_after_bm25(state: AdaptiveRagState) -> str:
        return "generate" if state["evidence_score"] >= score_threshold else "hybrid"

    def retrieve_hybrid_node(state: AdaptiveRagState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        try:
            docs = run_hybrid(query, bm25_retriever, vector_retriever)
        except Exception:
            docs = []
        return {
            "retrieved_docs": docs,
            "retrieval_mode": "hybrid",
            "retrieval_path": ["→ 升级 Hybrid"],
        }

    def evaluate_hybrid_node(state: AdaptiveRagState) -> dict:
        score = score_docs(state["retrieved_docs"], state["question"])
        return {"evidence_score": score}

    def route_after_hybrid(state: AdaptiveRagState) -> str:
        return "generate" if state["evidence_score"] >= score_threshold else "fallback"

    def generate_node(state: AdaptiveRagState) -> dict:
        docs = state["retrieved_docs"]
        context = _format_docs(docs)
        sources = _extract_sources(docs)
        messages = _build_messages(
            state["question"],
            context,
            state.get("chat_history", []),
            state.get("summary", ""),
        )
        try:
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            answer = "回答生成失败，请稍后重试。"
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_path": ["→ 生成答案"],
        }

    def fallback_node(state: AdaptiveRagState) -> dict:
        return {
            "answer": "未检索到足够可靠的政策依据，暂时无法回答该问题。",
            "sources": [],
            "retrieval_mode": "fallback",
            "retrieval_path": ["→ Fallback"],
        }

    # ── Graph wiring ──────────────────────────────────────────────────────────

    workflow = StateGraph(AdaptiveRagState)

    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve_bm25", retrieve_bm25_node)
    workflow.add_node("evaluate_bm25", evaluate_bm25_node)
    workflow.add_node("retrieve_hybrid", retrieve_hybrid_node)
    workflow.add_node("evaluate_hybrid", evaluate_hybrid_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve_bm25")
    workflow.add_edge("retrieve_bm25", "evaluate_bm25")
    workflow.add_conditional_edges(
        "evaluate_bm25",
        route_after_bm25,
        {"generate": "generate", "hybrid": "retrieve_hybrid"},
    )
    workflow.add_edge("retrieve_hybrid", "evaluate_hybrid")
    workflow.add_conditional_edges(
        "evaluate_hybrid",
        route_after_hybrid,
        {"generate": "generate", "fallback": "fallback"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


# ── Runner ────────────────────────────────────────────────────────────────────

def run_adaptive_rag(
    graph: Any,
    question: str,
    chat_history: list[dict] | None = None,
    summary: str = "",
) -> AdaptiveRagRunResult:
    initial_state: AdaptiveRagState = {
        "question": question,
        "rewritten_query": "",
        "retrieved_docs": [],
        "evidence_score": 0.0,
        "retrieval_mode": "bm25",
        "answer": "",
        "sources": [],
        "retrieval_path": [],
        "chat_history": chat_history or [],
        "summary": summary,
    }

    final_state = graph.invoke(initial_state)

    return AdaptiveRagRunResult(
        answer=final_state["answer"],
        sources=final_state.get("sources", []),
        retrieval_mode=final_state.get("retrieval_mode", "bm25"),
        retrieval_path=final_state.get("retrieval_path", []),
        evidence_score=final_state.get("evidence_score", 0.0),
    )
