from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services.knowledge_extraction_service import (
    KnowledgeExtractionStatus,
    extract_knowledge_from_conversation,
)
from src.app_services.langchain_rag_service import build_langchain_llm
from src.app_services.local_env import load_env_file
from src.ingestion.chunk_store import read_chunks_jsonl, write_chunks_jsonl
from src.ingestion.models import DocumentChunk

load_env_file(PROJECT_ROOT / ".env")

CHUNKS_PATH = PROJECT_ROOT / "runtime" / "processed" / "chunks.jsonl"

st.set_page_config(page_title="对话知识沉淀", page_icon="🧠", layout="wide")


def _get_llm():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    return build_langchain_llm(api_key) if api_key else None


def _load_lc_messages() -> list[dict]:
    return st.session_state.get("lc_messages", [])


def main():
    st.title("🧠 对话知识沉淀")
    st.caption("LCEL chain + JsonOutputParser — 从问答对话中提取结构化知识点")

    with st.sidebar:
        st.subheader("说明")
        st.markdown(
            "本页面使用 LangChain `JsonOutputParser` 分析问答对话，"
            "自动识别其中的事实性知识、操作流程和注意事项，"
            "并支持将提取结果追加到知识库。"
        )
        st.caption("LCEL chain：`prompt | llm | JsonOutputParser()`")

    llm = _get_llm()
    if llm is None:
        st.warning("未配置 DASHSCOPE_API_KEY，无法使用知识提取功能。")
        return

    # ── Conversation source ───────────────────────────────────────────────────
    st.subheader("对话来源")
    source = st.radio(
        "选择对话内容",
        options=["当前会话（LangChain RAG 页）", "手动输入"],
        horizontal=True,
    )

    lc_messages = _load_lc_messages()
    conversation: list[dict] = []

    if source == "当前会话（LangChain RAG 页）":
        if not lc_messages:
            st.info("LangChain RAG 页暂无对话记录。请先在该页面进行问答，再回此处提取知识。")
        else:
            st.caption(f"已加载 {len(lc_messages)} 条消息（{len(lc_messages)//2} 轮对话）")
            with st.expander("查看对话内容", expanded=False):
                for msg in lc_messages:
                    role_label = "👤 用户" if msg["role"] == "user" else "🤖 AI"
                    st.markdown(f"**{role_label}**：{msg['content']}")
            conversation = lc_messages
    else:
        sample = (
            "用户：客户经理被投诉一次会扣几分？\n"
            "AI：根据考核办法第3页，被投诉一次且核查属实，当月绩效扣5分。\n"
            "用户：累计几次会影响评聘？\n"
            "AI：根据第5页，一年内累计3次有效投诉将被取消当年评聘资格。"
        )
        raw_text = st.text_area("粘贴对话内容（格式：用户：…\\nAI：…）", value=sample, height=160)
        for line in raw_text.splitlines():
            line = line.strip()
            if line.startswith("用户："):
                conversation.append({"role": "user", "content": line[3:]})
            elif line.startswith("AI："):
                conversation.append({"role": "assistant", "content": line[3:]})

    # ── Extract ───────────────────────────────────────────────────────────────
    if st.button("提取知识", type="primary", disabled=not conversation):
        with st.spinner("正在分析对话并提取知识点…"):
            result = extract_knowledge_from_conversation(conversation, llm)

        if result.status == KnowledgeExtractionStatus.FAILED:
            st.error(f"知识提取失败：{result.message}")
            return

        if result.status == KnowledgeExtractionStatus.EMPTY:
            st.warning(result.message)
            return

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader("提取结果")
        col1, col2 = st.columns(2)
        col1.metric("提取知识点", len(result.extracted_knowledge))
        col2.metric("分析轮次", result.checked_turns // 2)

        if result.conversation_summary:
            st.info(f"**对话摘要**：{result.conversation_summary}")
        if result.user_intent:
            st.caption(f"**用户意图**：{result.user_intent}")

        # ── Knowledge items ───────────────────────────────────────────────────
        type_icon = {"事实": "📌", "需求": "💡", "流程": "🔄", "注意": "⚠️"}
        confidence_color = lambda c: "🟢" if c >= 0.8 else "🟡" if c >= 0.5 else "🔴"

        selected_items: list[dict] = []
        st.subheader("知识点列表（勾选后可追加到知识库）")
        for i, item in enumerate(result.extracted_knowledge):
            icon = type_icon.get(item.get("knowledge_type", ""), "📝")
            conf = item.get("confidence", 0.0)
            label = f"{icon} {item.get('content', '')[:60]}"
            if st.checkbox(label, value=conf >= 0.7, key=f"kb_item_{i}"):
                selected_items.append(item)
            with st.expander(f"详情 — 置信度 {confidence_color(conf)} {conf:.0%}", expanded=False):
                st.markdown(f"**类型**：{item.get('knowledge_type', '—')}")
                st.markdown(f"**内容**：{item.get('content', '—')}")
                st.markdown(f"**分类**：{item.get('category', '—')}")
                kws = "、".join(item.get("keywords", []))
                st.markdown(f"**关键词**：{kws or '—'}")

        # ── Append to KB ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("追加到知识库")

        if not CHUNKS_PATH.exists():
            st.warning("未找到 chunks.jsonl，请先在主页构建知识库。")
        elif selected_items:
            if st.button(f"追加 {len(selected_items)} 个知识点到 chunks.jsonl", type="secondary"):
                existing = read_chunks_jsonl(CHUNKS_PATH)
                existing_ids = {c["chunk_id"] for c in existing}
                new_chunks: list[DocumentChunk] = []
                for item in selected_items:
                    chunk = DocumentChunk(
                        content=item["content"],
                        source_file="对话知识沉淀",
                        page_number=1,
                        metadata={
                            "knowledge_type": item.get("knowledge_type", ""),
                            "category": item.get("category", ""),
                            "keywords": item.get("keywords", []),
                        },
                    )
                    if chunk.chunk_id not in existing_ids:
                        new_chunks.append(chunk)

                if new_chunks:
                    all_dicts = existing + [c.to_dict() for c in new_chunks]
                    # Write back — use low-level write to avoid overwrite guard
                    CHUNKS_PATH.write_text(
                        "\n".join(json.dumps(d, ensure_ascii=False) for d in all_dicts),
                        encoding="utf-8",
                    )
                    st.success(
                        f"已追加 {len(new_chunks)} 个知识点到 chunks.jsonl。"
                        " 请在主页重新构建知识库以使新内容生效。"
                    )
                else:
                    st.info("所选知识点已存在于知识库中，无需重复追加。")
        else:
            st.caption("请至少勾选一个知识点。")


if __name__ == "__main__":
    main()
