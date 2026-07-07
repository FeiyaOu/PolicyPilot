from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class LangChainKnowledgeBaseStatus(StrEnum):
    READY = "ready"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True)
class LangChainKnowledgeBaseResult:
    status: LangChainKnowledgeBaseStatus
    chunk_count: int
    vectorstore: Any | None
    index_dir: Path | None
    message: str


def build_langchain_knowledge_base(
    chunks: list[dict[str, Any]],
    index_dir: str | Path,
    embeddings: Any,
    faiss_cls: Any = None,
) -> LangChainKnowledgeBaseResult:
    if not chunks:
        return LangChainKnowledgeBaseResult(
            status=LangChainKnowledgeBaseStatus.EMPTY,
            chunk_count=0,
            vectorstore=None,
            index_dir=None,
            message="chunks 为空，无法构建 LangChain 向量索引。",
        )

    faiss = faiss_cls or _load_faiss()
    output_dir = Path(index_dir)

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
        }
        for chunk in chunks
    ]

    vectorstore = faiss.from_texts(texts, embeddings, metadatas=metadatas)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))

    return LangChainKnowledgeBaseResult(
        status=LangChainKnowledgeBaseStatus.READY,
        chunk_count=len(chunks),
        vectorstore=vectorstore,
        index_dir=output_dir,
        message=f"LangChain FAISS 索引构建完成：{len(chunks)} 个 chunk。",
    )


def load_langchain_knowledge_base(
    index_dir: str | Path,
    embeddings: Any,
    faiss_cls: Any = None,
) -> LangChainKnowledgeBaseResult:
    faiss = faiss_cls or _load_faiss()
    path = Path(index_dir)

    if not (path / "index.faiss").exists():
        return LangChainKnowledgeBaseResult(
            status=LangChainKnowledgeBaseStatus.EMPTY,
            chunk_count=0,
            vectorstore=None,
            index_dir=None,
            message="未找到 LangChain FAISS 索引，请先构建。",
        )

    vectorstore = faiss.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return LangChainKnowledgeBaseResult(
        status=LangChainKnowledgeBaseStatus.READY,
        chunk_count=-1,
        vectorstore=vectorstore,
        index_dir=path,
        message="已加载 LangChain FAISS 索引。",
    )


def _load_faiss() -> Any:
    from langchain_community.vectorstores import FAISS

    return FAISS
