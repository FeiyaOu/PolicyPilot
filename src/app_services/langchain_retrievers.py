from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# ── Query Rewrite ─────────────────────────────────────────────────────────────

_REWRITE_PROMPT = PromptTemplate.from_template(
    "你是一个擅长将用户问题改写为更精准检索查询的助手。\n"
    "请将以下问题改写成更利于检索的查询，只输出改写后的查询，不要其他内容。\n\n"
    "原始问题：{question}"
)


def build_query_rewrite_chain(llm: Any):
    """LCEL chain: {question} → rewritten query string."""
    return _REWRITE_PROMPT | llm | StrOutputParser()


# ── MultiQuery ────────────────────────────────────────────────────────────────

_MULTIQUERY_PROMPT = PromptTemplate.from_template(
    "为以下问题生成 {num_variants} 个不同角度的检索查询变体，每行一个，不要编号：\n\n"
    "原始问题：{question}"
)


def run_multiquery(
    question: str,
    retriever: Any,
    llm: Any,
    num_variants: int = 3,
    k: int = 4,
) -> list[Document]:
    """Generate query variants with LLM, run each through retriever, deduplicate."""
    variant_chain = _MULTIQUERY_PROMPT | llm | StrOutputParser()

    try:
        variants_text = variant_chain.invoke(
            {"question": question, "num_variants": num_variants}
        )
        variants = [q.strip() for q in variants_text.split("\n") if q.strip()]
    except Exception:
        variants = []

    all_queries = [question] + variants[:num_variants]
    seen: set[str] = set()
    results: list[Document] = []

    for query in all_queries:
        try:
            docs = retriever.invoke(query)
        except Exception:
            continue
        for doc in docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                results.append(doc)

    return results[:k * 2]  # return generous window; caller can slice


# ── Hybrid (Reciprocal Rank Fusion) ──────────────────────────────────────────

def run_hybrid(
    question: str,
    bm25_retriever: Any,
    vector_retriever: Any,
    k: int = 4,
    rrf_constant: int = 60,
) -> list[Document]:
    """BM25 + vector retrieval merged with Reciprocal Rank Fusion."""
    try:
        bm25_docs = bm25_retriever.invoke(question)
    except Exception:
        bm25_docs = []
    try:
        vec_docs = vector_retriever.invoke(question)
    except Exception:
        vec_docs = []

    return _reciprocal_rank_fusion([bm25_docs, vec_docs], k=k, rrf_constant=rrf_constant)


def _reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    k: int = 4,
    rrf_constant: int = 60,
) -> list[Document]:
    scores: dict[str, dict] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (rrf_constant + rank + 1)

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [entry["doc"] for entry in ranked[:k]]


# ── BM25 Retriever builder ────────────────────────────────────────────────────

def build_bm25_retriever(chunks: list[dict], k: int = 4) -> Any:
    """Build a LangChain BM25Retriever from a list of chunk dicts."""
    from langchain_community.retrievers import BM25Retriever

    docs = [
        Document(
            page_content=chunk["content"],
            metadata={
                "chunk_id": chunk.get("chunk_id", ""),
                "source_file": chunk.get("source_file", ""),
                "page_number": chunk.get("page_number"),
            },
        )
        for chunk in chunks
    ]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever
