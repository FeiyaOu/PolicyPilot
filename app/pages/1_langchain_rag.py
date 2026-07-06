from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services.langchain_knowledge_base import (
    LangChainKnowledgeBaseStatus,
    build_langchain_knowledge_base,
    load_langchain_knowledge_base,
)
from src.app_services.langchain_rag_service import (
    LangChainRagService,
    LangChainRagStatus,
    build_langchain_llm,
    should_compress_history,
    split_history_for_compression,
    summarize_history,
)
from src.app_services.local_env import load_env_file
from src.ingestion.chunk_store import read_chunks_jsonl

load_env_file(PROJECT_ROOT / ".env")

CHUNKS_PATH = PROJECT_ROOT / "runtime" / "processed" / "chunks.jsonl"
LC_INDEX_DIR = PROJECT_ROOT / "runtime" / "vector_index_lc"

st.set_page_config(page_title="LangChain RAG", page_icon="🦜", layout="wide")


# ── cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import DashScopeEmbeddings

    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    )


@st.cache_resource
def get_langchain_kb():
    embeddings = get_embeddings()

    # Try loading existing LangChain index first
    result = load_langchain_knowledge_base(LC_INDEX_DIR, embeddings)
    if result.status == LangChainKnowledgeBaseStatus.READY:
        return result

    # Fall back to building from chunks.jsonl
    if not CHUNKS_PATH.exists():
        return result  # EMPTY — user needs to build V1 KB first

    chunks = read_chunks_jsonl(CHUNKS_PATH)
    return build_langchain_knowledge_base(chunks, LC_INDEX_DIR, embeddings)


def get_service() -> LangChainRagService | None:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return None
    kb = get_langchain_kb()
    if kb.status != LangChainKnowledgeBaseStatus.READY:
        return None
    retriever = kb.vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_langchain_llm(api_key)
    return LangChainRagService(retriever=retriever, llm=llm)


# ── session state ─────────────────────────────────────────────────────────────

def _init_session():
    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages = []      # list[dict] role/content
    if "lc_summary" not in st.session_state:
        st.session_state.lc_summary = ""
    if "lc_last_sources" not in st.session_state:
        st.session_state.lc_last_sources = []


def _maybe_compress(llm):
    """Compress history into summary if it exceeds the character threshold."""
    if not should_compress_history(st.session_state.lc_messages):
        return
    to_compress, to_keep = split_history_for_compression(st.session_state.lc_messages)
    if to_compress:
        st.session_state.lc_summary = summarize_history(
            to_compress, st.session_state.lc_summary, llm
        )
        st.session_state.lc_messages = to_keep


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("🦜 LangChain RAG")
        st.caption("V2 — 标准向量检索 + Memory")

        st.subheader("知识库状态")
        kb = get_langchain_kb()
        if kb.status == LangChainKnowledgeBaseStatus.READY:
            st.success(kb.message)
            if st.button("重新构建 LangChain 索引"):
                get_langchain_kb.clear()
                st.rerun()
        else:
            st.warning(kb.message)
            st.caption("请先在「PolicyPilot RAG」主页构建知识库。")

        st.subheader("Memory 状态")
        turn_count = len(st.session_state.get("lc_messages", [])) // 2
        has_summary = bool(st.session_state.get("lc_summary", ""))
        st.metric("对话轮次", turn_count)
        if has_summary:
            st.info("历史已压缩为摘要")
            with st.expander("查看摘要"):
                st.write(st.session_state.lc_summary)

        if st.button("清除对话"):
            st.session_state.lc_messages = []
            st.session_state.lc_summary = ""
            st.session_state.lc_last_sources = []
            st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _init_session()
    render_sidebar()

    st.title("🦜 LangChain RAG 制度问答")
    st.caption("使用 LangChain 向量检索 + ConversationSummaryBuffer Memory")

    service = get_service()
    if service is None:
        kb = get_langchain_kb()
        if kb.status != LangChainKnowledgeBaseStatus.READY:
            st.warning("知识库未就绪，请先在主页构建知识库。")
        else:
            st.warning("未配置 DASHSCOPE_API_KEY，无法使用 LangChain RAG。")
        return

    # Render existing conversation
    for msg in st.session_state.lc_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show last retrieval details
    if st.session_state.lc_last_sources:
        with st.expander("最新检索详情", expanded=False):
            for i, src in enumerate(st.session_state.lc_last_sources, 1):
                page_label = f"第 {src['page_number']} 页" if src["page_number"] else "页码未知"
                st.markdown(f"**{i}. {src['source_file']}** — {page_label}")
                st.caption(src["content"][:200])

    # Chat input
    if question := st.chat_input("请输入制度问题…"):
        st.session_state.lc_messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("检索中…"):
                result = service.answer(
                    question=question,
                    chat_history=st.session_state.lc_messages[:-1],
                    summary=st.session_state.lc_summary,
                )

            if result.status == LangChainRagStatus.ANSWERED:
                st.markdown(result.answer)
            else:
                st.warning(result.answer or "回答生成失败，请稍后重试。")

        st.session_state.lc_messages.append(
            {"role": "assistant", "content": result.answer}
        )
        st.session_state.lc_last_sources = result.sources

        # Compress history if needed
        _maybe_compress(service.llm)

        st.rerun()


if __name__ == "__main__":
    main()
