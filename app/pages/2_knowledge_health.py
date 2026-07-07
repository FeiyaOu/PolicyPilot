from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services.knowledge_health_service import (
    DEFAULT_QUERIES,
    KnowledgeHealthStatus,
    check_knowledge_health,
)
from src.app_services.langchain_knowledge_base import (
    LangChainKnowledgeBaseStatus,
    load_langchain_knowledge_base,
)
from src.app_services.langchain_rag_service import build_langchain_llm
from src.app_services.local_env import load_env_file

load_env_file(PROJECT_ROOT / ".env")

LC_INDEX_DIR = PROJECT_ROOT / "runtime" / "vector_index_lc"

st.set_page_config(page_title="知识库健康检查", page_icon="🏥", layout="wide")


@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    )


def _get_retriever():
    kb = load_langchain_knowledge_base(LC_INDEX_DIR, get_embeddings())
    if kb.status != LangChainKnowledgeBaseStatus.READY:
        return None
    return kb.vectorstore.as_retriever(search_kwargs={"k": 3})


def _get_llm():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    return build_langchain_llm(api_key) if api_key else None


def main():
    st.title("🏥 知识库健康检查")
    st.caption("LCEL chain + JsonOutputParser — 检测知识库覆盖率与知识空白")

    with st.sidebar:
        st.subheader("说明")
        st.markdown(
            "本页面使用 LangChain `JsonOutputParser` 对知识库进行结构化分析，"
            "识别覆盖率不足的领域并给出补充建议。"
        )
        st.caption("LCEL chain：`prompt | llm | JsonOutputParser()`")

    retriever = _get_retriever()
    llm = _get_llm()

    if retriever is None:
        st.warning("LangChain 向量索引未就绪，请先在「LangChain RAG」页面初始化知识库。")
        return
    if llm is None:
        st.warning("未配置 DASHSCOPE_API_KEY，无法执行健康检查。")
        return

    # ── Query editor ──────────────────────────────────────────────────────────
    st.subheader("测试问题")
    st.caption("编辑或使用默认的银行政策测试问题集")

    raw_queries = st.text_area(
        "每行一个测试问题",
        value="\n".join(DEFAULT_QUERIES),
        height=160,
    )
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()]
    st.caption(f"共 {len(queries)} 个测试问题")

    # ── Run check ─────────────────────────────────────────────────────────────
    if st.button("开始健康检查", type="primary", disabled=not queries):
        with st.spinner(f"正在检索 {len(queries)} 个问题并分析覆盖情况…"):
            result = check_knowledge_health(
                queries=queries,
                retriever=retriever,
                llm=llm,
            )

        if result.status == KnowledgeHealthStatus.FAILED:
            st.error(f"健康检查失败：{result.message}")
            return

        # ── Score ─────────────────────────────────────────────────────────────
        st.subheader("检查结果")
        col1, col2 = st.columns(2)
        score_pct = int(result.coverage_score * 100)
        col1.metric("覆盖率评分", f"{score_pct}%")
        col2.metric("检测问题数", result.checked_queries)

        if score_pct >= 75:
            st.success(f"知识库覆盖率良好（{score_pct}%）")
        elif score_pct >= 50:
            st.warning(f"知识库覆盖率一般（{score_pct}%），建议补充以下内容")
        else:
            st.error(f"知识库覆盖率不足（{score_pct}%），需要大量补充")

        # ── Analysis ──────────────────────────────────────────────────────────
        if result.completeness_analysis:
            st.info(result.completeness_analysis)

        # ── Missing knowledge ─────────────────────────────────────────────────
        if result.missing_knowledge:
            st.subheader(f"知识空白（{len(result.missing_knowledge)} 项）")
            importance_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}
            for item in result.missing_knowledge:
                icon = importance_icon.get(item.get("importance", ""), "⚪")
                with st.expander(
                    f"{icon} {item.get('query', '—')} — {item.get('missing_aspect', '')}",
                    expanded=item.get("importance") == "高",
                ):
                    st.markdown(f"**重要性**：{item.get('importance', '—')}")
                    st.markdown(f"**缺失方面**：{item.get('missing_aspect', '—')}")
                    st.markdown(f"**建议补充**：{item.get('suggested_content', '—')}")
        else:
            st.success("未发现明显知识空白 🎉")


if __name__ == "__main__":
    main()
