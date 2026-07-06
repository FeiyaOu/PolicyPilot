from __future__ import annotations

from unittest.mock import MagicMock

from src.app_services.langchain_rag_service import (
    LangChainRagService,
    LangChainRagStatus,
    should_compress_history,
    split_history_for_compression,
    summarize_history,
)


# ── helpers ──────────────────────────────────────────────────────────────────

class FakeDoc:
    def __init__(self, content: str, source_file: str = "policy.pdf", page: int = 1):
        self.page_content = content
        self.metadata = {"source_file": source_file, "page_number": page}


def _fake_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=response_text)
    return llm


def _fake_retriever(docs: list) -> MagicMock:
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    return retriever


# ── LangChainRagService ───────────────────────────────────────────────────────

def test_service_returns_answer_with_sources():
    docs = [FakeDoc("客户经理被投诉一次会影响评聘。", page=3)]
    svc = LangChainRagService(
        retriever=_fake_retriever(docs),
        llm=_fake_llm("根据第3页，投诉会影响评聘。"),
    )

    result = svc.answer("投诉影响评聘吗？", chat_history=[])

    assert result.status == LangChainRagStatus.ANSWERED
    assert "投诉" in result.answer
    assert result.retrieval_count == 1
    assert result.sources[0]["source_file"] == "policy.pdf"
    assert result.sources[0]["page_number"] == 3


def test_service_returns_no_context_when_retriever_empty():
    svc = LangChainRagService(
        retriever=_fake_retriever([]),
        llm=_fake_llm(""),
    )

    result = svc.answer("投诉影响评聘吗？", chat_history=[])

    assert result.status == LangChainRagStatus.NO_CONTEXT
    assert result.retrieval_count == 0
    assert result.sources == []


def test_service_includes_chat_history_in_llm_call():
    docs = [FakeDoc("考核规则内容。")]
    llm = _fake_llm("答案。")
    svc = LangChainRagService(retriever=_fake_retriever(docs), llm=llm)

    chat_history = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个回答"},
    ]
    svc.answer("第二个问题", chat_history=chat_history)

    messages_passed = llm.invoke.call_args[0][0]
    all_content = " ".join(
        m.content for m in messages_passed if hasattr(m, "content")
    )
    assert "第一个问题" in all_content
    assert "第一个回答" in all_content


def test_service_injects_summary_into_messages_when_provided():
    docs = [FakeDoc("内容。")]
    llm = _fake_llm("答案。")
    svc = LangChainRagService(retriever=_fake_retriever(docs), llm=llm)

    svc.answer("问题", chat_history=[], summary="之前对话的摘要内容")

    messages_passed = llm.invoke.call_args[0][0]
    all_content = " ".join(
        m.content for m in messages_passed if hasattr(m, "content")
    )
    assert "之前对话的摘要内容" in all_content
    assert svc.answer("问题", chat_history=[], summary="摘要").used_summary is True


def test_service_returns_failed_when_llm_raises():
    docs = [FakeDoc("内容。")]
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("API error")
    svc = LangChainRagService(retriever=_fake_retriever(docs), llm=llm)

    result = svc.answer("问题", chat_history=[])

    assert result.status == LangChainRagStatus.FAILED


def test_service_deduplicates_sources_from_same_page():
    docs = [
        FakeDoc("内容A。", source_file="p.pdf", page=2),
        FakeDoc("内容B。", source_file="p.pdf", page=2),
        FakeDoc("内容C。", source_file="p.pdf", page=5),
    ]
    svc = LangChainRagService(retriever=_fake_retriever(docs), llm=_fake_llm("答案"))

    result = svc.answer("问题", chat_history=[])

    assert len(result.sources) == 2
    pages = {s["page_number"] for s in result.sources}
    assert pages == {2, 5}


# ── Memory helpers ────────────────────────────────────────────────────────────

def test_should_compress_returns_false_when_under_limit():
    history = [
        {"role": "user", "content": "短问题"},
        {"role": "assistant", "content": "短回答"},
    ]
    assert should_compress_history(history, max_chars=2000) is False


def test_should_compress_returns_true_when_over_limit():
    history = [{"role": "user", "content": "x" * 1001}] * 3
    assert should_compress_history(history, max_chars=2000) is True


def test_split_history_keeps_recent_and_returns_older_to_compress():
    history = [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"},
        {"role": "user", "content": "最近的问题"},
        {"role": "assistant", "content": "最近的回答"},
    ]
    to_compress, to_keep = split_history_for_compression(history, keep_chars=30)

    assert len(to_keep) > 0
    assert len(to_compress) + len(to_keep) == len(history)
    # most recent messages end up in to_keep
    assert to_keep[-1]["content"] == "最近的回答"


def test_split_history_returns_all_to_keep_when_short():
    history = [{"role": "user", "content": "短"}]
    to_compress, to_keep = split_history_for_compression(history, keep_chars=5000)

    assert to_compress == []
    assert to_keep == history


def test_summarize_history_calls_llm_and_returns_content():
    llm = _fake_llm("摘要内容")
    messages = [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
    ]

    result = summarize_history(messages, existing_summary="", llm=llm)

    assert result == "摘要内容"
    assert llm.invoke.called


def test_summarize_history_includes_existing_summary():
    llm = _fake_llm("新摘要")
    messages = [{"role": "user", "content": "问题"}]

    summarize_history(messages, existing_summary="旧摘要", llm=llm)

    prompt_content = llm.invoke.call_args[0][0][0].content
    assert "旧摘要" in prompt_content


def test_summarize_history_returns_existing_when_no_messages():
    llm = _fake_llm("不应被调用")
    result = summarize_history([], existing_summary="保持原样", llm=llm)

    assert result == "保持原样"
    assert not llm.invoke.called
