from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.ingestion.chunk_store import read_chunks_jsonl
from src.retrieval.vector_index import EmbeddingProvider, build_faiss_vector_index, save_faiss_vector_index


class VectorIndexBuildStatus(StrEnum):
    BUILT = "built"
    SKIPPED = "skipped"
    MISSING_CHUNKS = "missing_chunks"
    EMPTY_CHUNKS = "empty_chunks"
    FAILED = "failed"


@dataclass(frozen=True)
class VectorIndexBuildPageResult:
    status: VectorIndexBuildStatus
    chunk_count: int
    index_dir: Path | None
    embedding_model: str | None
    message: str


def build_vector_index_from_chunks(
    chunks_path: str | Path,
    index_dir: str | Path,
    embedding_provider: EmbeddingProvider | None,
    embedding_model: str | None,
) -> VectorIndexBuildPageResult:
    if embedding_provider is None:
        return VectorIndexBuildPageResult(
            status=VectorIndexBuildStatus.SKIPPED,
            chunk_count=0,
            index_dir=None,
            embedding_model=embedding_model,
            message="未配置 Embedding Provider，已跳过 FAISS 向量索引构建。",
        )

    path = Path(chunks_path)
    if not path.exists():
        return VectorIndexBuildPageResult(
            status=VectorIndexBuildStatus.MISSING_CHUNKS,
            chunk_count=0,
            index_dir=None,
            embedding_model=embedding_model,
            message="未找到 chunks.jsonl，无法构建 FAISS 向量索引。",
        )

    chunk_records = read_chunks_jsonl(path)
    if not chunk_records:
        return VectorIndexBuildPageResult(
            status=VectorIndexBuildStatus.EMPTY_CHUNKS,
            chunk_count=0,
            index_dir=None,
            embedding_model=embedding_model,
            message="chunks.jsonl 为空，无法构建 FAISS 向量索引。",
        )

    try:
        vector_index = build_faiss_vector_index(chunk_records, embedding_provider)
        output_dir = save_faiss_vector_index(vector_index, index_dir)
    except Exception as error:
        return VectorIndexBuildPageResult(
            status=VectorIndexBuildStatus.FAILED,
            chunk_count=len(chunk_records),
            index_dir=None,
            embedding_model=embedding_model,
            message=_build_embedding_failure_message(error),
        )

    model_label = embedding_model or "unknown"
    return VectorIndexBuildPageResult(
        status=VectorIndexBuildStatus.BUILT,
        chunk_count=len(chunk_records),
        index_dir=output_dir,
        embedding_model=embedding_model,
        message=f"FAISS 向量索引构建完成：{len(chunk_records)} 个 chunk，模型 {model_label}。",
    )


def _build_embedding_failure_message(error: Exception) -> str:
    error_text = str(error)
    if "400" in error_text:
        return "FAISS 向量索引构建失败：Embedding 服务返回错误 400。请检查模型、维度、API Key 权限或输入文本。"

    return "FAISS 向量索引构建失败：Embedding 服务调用失败。请检查模型、维度、API Key 权限或网络连接。"