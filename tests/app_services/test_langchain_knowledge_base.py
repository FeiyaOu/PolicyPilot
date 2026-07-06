from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.app_services.langchain_knowledge_base import (
    LangChainKnowledgeBaseStatus,
    LangChainKnowledgeBaseResult,
    build_langchain_knowledge_base,
)


SAMPLE_CHUNKS = [
    {
        "chunk_id": "policy-p1-c0",
        "content": "客户经理被投诉一次会影响评聘。",
        "source_file": "policy.pdf",
        "page_number": 1,
        "metadata": {},
    },
    {
        "chunk_id": "policy-p2-c0",
        "content": "投诉记录需归档保存备查。",
        "source_file": "policy.pdf",
        "page_number": 2,
        "metadata": {},
    },
]


class FakeVectorStore:
    """Minimal FAISS-like stub returned by FAISS.from_texts."""

    def __init__(self):
        self.saved_path = None

    def save_local(self, path: str) -> None:
        self.saved_path = path

    def as_retriever(self, **kwargs):
        return MagicMock()


def _make_fake_faiss(fake_store: FakeVectorStore):
    """Return a FAISS class-level mock whose from_texts returns fake_store."""
    faiss_cls = MagicMock()
    faiss_cls.from_texts.return_value = fake_store
    faiss_cls.load_local.return_value = fake_store
    return faiss_cls


def test_build_langchain_knowledge_base_returns_ready_when_chunks_exist(tmp_path):
    fake_store = FakeVectorStore()
    fake_faiss = _make_fake_faiss(fake_store)
    fake_embeddings = MagicMock()

    result = build_langchain_knowledge_base(
        chunks=SAMPLE_CHUNKS,
        index_dir=tmp_path / "vector_index_lc",
        embeddings=fake_embeddings,
        faiss_cls=fake_faiss,
    )

    assert result.status == LangChainKnowledgeBaseStatus.READY
    assert result.chunk_count == 2
    assert result.vectorstore is fake_store
    assert result.index_dir == tmp_path / "vector_index_lc"
    assert fake_faiss.from_texts.called


def test_build_langchain_knowledge_base_passes_texts_and_metadata(tmp_path):
    fake_store = FakeVectorStore()
    fake_faiss = _make_fake_faiss(fake_store)
    fake_embeddings = MagicMock()

    build_langchain_knowledge_base(
        chunks=SAMPLE_CHUNKS,
        index_dir=tmp_path / "vector_index_lc",
        embeddings=fake_embeddings,
        faiss_cls=fake_faiss,
    )

    call_kwargs = fake_faiss.from_texts.call_args
    texts = call_kwargs[0][0]
    metadatas = call_kwargs[1]["metadatas"]

    assert texts == ["客户经理被投诉一次会影响评聘。", "投诉记录需归档保存备查。"]
    assert metadatas[0]["source_file"] == "policy.pdf"
    assert metadatas[0]["page_number"] == 1
    assert metadatas[0]["chunk_id"] == "policy-p1-c0"


def test_build_langchain_knowledge_base_saves_index_to_disk(tmp_path):
    fake_store = FakeVectorStore()
    fake_faiss = _make_fake_faiss(fake_store)
    fake_embeddings = MagicMock()
    index_dir = tmp_path / "vector_index_lc"

    build_langchain_knowledge_base(
        chunks=SAMPLE_CHUNKS,
        index_dir=index_dir,
        embeddings=fake_embeddings,
        faiss_cls=fake_faiss,
    )

    assert fake_store.saved_path == str(index_dir)


def test_build_langchain_knowledge_base_returns_empty_when_no_chunks(tmp_path):
    fake_faiss = _make_fake_faiss(FakeVectorStore())
    fake_embeddings = MagicMock()

    result = build_langchain_knowledge_base(
        chunks=[],
        index_dir=tmp_path / "vector_index_lc",
        embeddings=fake_embeddings,
        faiss_cls=fake_faiss,
    )

    assert result.status == LangChainKnowledgeBaseStatus.EMPTY
    assert result.chunk_count == 0
    assert result.vectorstore is None
    assert not fake_faiss.from_texts.called
