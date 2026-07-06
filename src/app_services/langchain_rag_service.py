from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


SYSTEM_PROMPT = (
    "你是银行内部制度问答助手，仅根据以下参考文档回答问题。"
    "回答时请引用来源文件名和页码。"
    "如文档中无相关信息，请说明无法在知识库中找到答案。"
)

SUMMARY_PROMPT = (
    "请将以下对话历史压缩成简洁摘要，保留所有重要的政策信息和问答要点。"
    "如有已有摘要，请将新内容合并进去。"
)

MAX_HISTORY_CHARS: int = 2000
KEEP_RECENT_CHARS: int = 800


class LangChainRagStatus(StrEnum):
    ANSWERED = "answered"
    NO_CONTEXT = "no_context"
    FAILED = "failed"


@dataclass(frozen=True)
class LangChainRagResult:
    answer: str
    sources: list[dict]
    retrieval_count: int
    used_summary: bool
    status: LangChainRagStatus


@dataclass(frozen=True)
class LangChainRagService:
    retriever: Any
    llm: Any

    def answer(
        self,
        question: str,
        chat_history: list[dict],
        summary: str = "",
    ) -> LangChainRagResult:
        try:
            docs = self.retriever.invoke(question)
        except Exception:
            return LangChainRagResult(
                answer="",
                sources=[],
                retrieval_count=0,
                used_summary=bool(summary),
                status=LangChainRagStatus.FAILED,
            )

        if not docs:
            return LangChainRagResult(
                answer="未检索到相关政策内容，无法回答该问题。",
                sources=[],
                retrieval_count=0,
                used_summary=bool(summary),
                status=LangChainRagStatus.NO_CONTEXT,
            )

        context = _format_docs(docs)
        sources = _extract_sources(docs)
        messages = _build_messages(question, context, chat_history, summary)

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
        except Exception:
            return LangChainRagResult(
                answer="",
                sources=sources,
                retrieval_count=len(docs),
                used_summary=bool(summary),
                status=LangChainRagStatus.FAILED,
            )

        return LangChainRagResult(
            answer=answer_text,
            sources=sources,
            retrieval_count=len(docs),
            used_summary=bool(summary),
            status=LangChainRagStatus.ANSWERED,
        )


# ── Memory helpers ────────────────────────────────────────────────────────────

def should_compress_history(
    chat_history: list[dict],
    max_chars: int = MAX_HISTORY_CHARS,
) -> bool:
    total = sum(len(m.get("content", "")) for m in chat_history)
    return total > max_chars


def split_history_for_compression(
    chat_history: list[dict],
    keep_chars: int = KEEP_RECENT_CHARS,
) -> tuple[list[dict], list[dict]]:
    if not chat_history:
        return [], []

    to_keep: list[dict] = []
    chars_accumulated = 0

    for message in reversed(chat_history):
        chars_accumulated += len(message.get("content", ""))
        if chars_accumulated <= keep_chars:
            to_keep.insert(0, message)
        else:
            break

    to_compress = chat_history[: len(chat_history) - len(to_keep)]
    return to_compress, to_keep


def summarize_history(
    messages_to_compress: list[dict],
    existing_summary: str,
    llm: Any,
) -> str:
    if not messages_to_compress:
        return existing_summary

    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}"
        for m in messages_to_compress
    )

    prompt_parts = [SUMMARY_PROMPT]
    if existing_summary:
        prompt_parts.append(f"\n已有摘要：\n{existing_summary}")
    prompt_parts.append(f"\n新对话内容：\n{history_text}")

    try:
        response = llm.invoke([HumanMessage(content="".join(prompt_parts))])
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        return existing_summary


# ── LLM builder ──────────────────────────────────────────────────────────────

def build_langchain_llm(api_key: str, model: str = "deepseek-v3") -> Any:
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi(model_name=model, dashscope_api_key=api_key)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _format_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        source = meta.get("source_file", "未知来源")
        page = meta.get("page_number", "?")
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        parts.append(f"[文档{i}] 来源：{source} 第{page}页\n{content}")
    return "\n\n".join(parts)


def _extract_sources(docs: list) -> list[dict]:
    sources: list[dict] = []
    seen: set[tuple] = set()
    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        source_file = meta.get("source_file", "未知来源")
        page_number = meta.get("page_number")
        key = (source_file, page_number)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source_file": source_file,
                    "page_number": page_number,
                    "content": doc.page_content if hasattr(doc, "page_content") else "",
                }
            )
    return sources


def _build_messages(
    question: str,
    context: str,
    chat_history: list[dict],
    summary: str,
) -> list:
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    if summary:
        messages.append(SystemMessage(content=f"历史对话摘要：\n{summary}"))

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=f"参考文档：\n{context}\n\n问题：{question}"))
    return messages
