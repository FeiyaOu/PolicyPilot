from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.app_services.retrieval_service import RetrievalMode, RetrievalService
from src.ingestion.chunk_store import read_chunks_jsonl
from src.retrieval.bm25 import Bm25Retriever
from src.retrieval.vector_index import EmbeddingProvider, load_faiss_vector_index


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
    vector_index_loaded: bool
    available_modes: tuple[RetrievalMode, ...]
    message: str


def load_knowledge_base(
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    vector_index_dir: str | Path | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> KnowledgeBaseLoadResult:
    path = Path(chunks_path)

    if not path.exists():
        return KnowledgeBaseLoadResult(
            status=KnowledgeBaseStatus.MISSING,
            chunk_count=0,
            retrieval_service=None,
            vector_index_loaded=False,
            available_modes=(),
            message="未找到知识库 chunks.jsonl，请先构建或加载知识库。",
        )

    chunk_records = read_chunks_jsonl(path)
    if not chunk_records:
        return KnowledgeBaseLoadResult(
            status=KnowledgeBaseStatus.EMPTY,
            chunk_count=0,
            retrieval_service=None,
            vector_index_loaded=False,
            available_modes=(),
            message="知识库 chunks.jsonl 为空，请重新构建知识库。",
        )

    vector_retriever = _load_vector_retriever(vector_index_dir, embedding_provider)
    vector_index_loaded = vector_retriever is not None
    available_modes = (RetrievalMode.BM25,)
    message = f"已加载 {len(chunk_records)} 个知识库 chunk。"
    if vector_index_loaded:
        available_modes = (RetrievalMode.BM25, RetrievalMode.VECTOR, RetrievalMode.HYBRID)
        message = f"已加载 {len(chunk_records)} 个知识库 chunk，并加载 FAISS 向量索引。"

    retrieval_service = RetrievalService(
        vector_retriever=vector_retriever,
        bm25_retriever=Bm25Retriever(chunk_records),
        embedding_provider=embedding_provider,
    )
    return KnowledgeBaseLoadResult(
        status=KnowledgeBaseStatus.READY,
        chunk_count=len(chunk_records),
        retrieval_service=retrieval_service,
        vector_index_loaded=vector_index_loaded,
        available_modes=available_modes,
        message=message,
    )


def _load_vector_retriever(
    vector_index_dir: str | Path | None,
    embedding_provider: EmbeddingProvider | None,
):
    if vector_index_dir is None or embedding_provider is None:
        return None

    index_dir = Path(vector_index_dir)
    if not (index_dir / "index.faiss").exists() or not (index_dir / "chunks.json").exists():
        return None

    return load_faiss_vector_index(index_dir)