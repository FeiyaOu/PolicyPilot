from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer_contract import AnswerResult, AnswerStatus
from src.app_services.demo_answer_service import DemoAnswerService, build_demo_answer_service


st.set_page_config(
    page_title="PolicyPilot RAG",
    page_icon="PolicyPilot",
    layout="wide",
)


@st.cache_resource
def get_answer_service() -> DemoAnswerService:
    return build_demo_answer_service()


def render_answer(answer: AnswerResult) -> None:
    if answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
        st.warning(answer.fallback_message)
    else:
        st.success(answer.answer_text)

    summary_columns = st.columns(3)
    summary_columns[0].metric("检索数量", answer.retrieval_summary["retrieved_count"])
    summary_columns[1].metric("入选证据", answer.retrieval_summary["selected_count"])
    summary_columns[2].metric("最高分数", _format_score(answer.retrieval_summary["max_score"]))

    citation_tab, evidence_tab, process_tab = st.tabs(["引用来源", "证据片段", "检索过程"])

    with citation_tab:
        if not answer.citations:
            st.caption("暂无引用来源")
        for citation in answer.citations:
            page_label = f"第 {citation.page_number} 页" if citation.page_number is not None else "页码未知"
            st.markdown(f"- **{citation.source_file}**, {page_label}")

    with evidence_tab:
        if not answer.contexts:
            st.caption("暂无入选证据")
        for index, context in enumerate(answer.contexts, start=1):
            with st.expander(f"证据 {index}: {context['source_file']} / 第 {context['page_number']} 页"):
                st.write(context["content"])
                st.caption(f"chunk_id: {context['chunk_id']} | score: {_format_score(context['score'])}")

    with process_tab:
        st.json(answer.retrieval_summary)


def _format_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def main() -> None:
    st.title("PolicyPilot RAG")
    st.caption("银行内部制度问答演示")

    with st.sidebar:
        top_k = st.slider("证据数量", min_value=1, max_value=4, value=2)
        min_score = st.slider("最低证据分数", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    question = st.text_area(
        "请输入制度问题",
        value="客户经理被投诉一次会影响评聘吗？",
        height=100,
    )

    if st.button("开始回答", type="primary"):
        if not question.strip():
            st.warning("请输入问题后再开始回答。")
            return

        answer = get_answer_service().answer(
            question=question,
            top_k=top_k,
            min_score=min_score,
        )
        render_answer(answer)


if __name__ == "__main__":
    main()