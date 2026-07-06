from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.app_services.retrieval_service import RetrievalService
from src.ingestion.chunk_store import read_chunks_jsonl
from src.retrieval.bm25 import Bm25Retriever


DEFAULT_CHUNKS_PATH = Path("runtime/processed/chunks.jsonl")


class KnowledgeBaseStatus(StrEnum):
    READY = "ready"
    MISSING = "missing"
    EMPTY = "empty"


@dataclass(frozen=True)
class KnowledgeBaseLoadResult:
    status: KnowledgeBaseStatus
    chunk_count: int
    retrieval_service: RetrievalService | None
    message: str


def load_knowledge_base(chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> KnowledgeBaseLoadResult:
    path = Path(chunks_path)

    if not path.exists():
        return KnowledgeBaseLoadResult(
            status=KnowledgeBaseStatus.MISSING,
            chunk_count=0,
            retrieval_service=None,
            message="未找到知识库 chunks.jsonl，请先构建或加载知识库。",
        )

    chunk_records = read_chunks_jsonl(path)
    if not chunk_records:
        return KnowledgeBaseLoadResult(
            status=KnowledgeBaseStatus.EMPTY,
            chunk_count=0,
            retrieval_service=None,
            message="知识库 chunks.jsonl 为空，请重新构建知识库。",
        )

    retrieval_service = RetrievalService(
        vector_retriever=None,
        bm25_retriever=Bm25Retriever(chunk_records),
        embedding_provider=None,
    )
    return KnowledgeBaseLoadResult(
        status=KnowledgeBaseStatus.READY,
        chunk_count=len(chunk_records),
        retrieval_service=retrieval_service,
        message=f"已加载 {len(chunk_records)} 个知识库 chunk。",
    )