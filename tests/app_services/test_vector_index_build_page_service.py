import json
from pathlib import Path

from src.app_services.vector_index_build_page_service import VectorIndexBuildStatus, build_vector_index_from_chunks


class FakeEmbeddingProvider:
    dimension = 2

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def write_chunks(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")


def test_build_vector_index_from_chunks_saves_faiss_index_and_metadata(tmp_path):
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    index_dir = tmp_path / "runtime" / "vector_index"
    write_chunks(
        chunks_path,
        [
            {
                "chunk_id": "chunk-1",
                "content": "客户经理被投诉一次会影响评聘。",
                "source_file": "policy-a.pdf",
                "page_number": 2,
                "metadata": {"chunk_index": 0},
            }
        ],
    )

    result = build_vector_index_from_chunks(
        chunks_path=chunks_path,
        index_dir=index_dir,
        embedding_provider=FakeEmbeddingProvider(),
        embedding_model="text-embedding-v4",
    )

    assert result.status == VectorIndexBuildStatus.BUILT
    assert result.chunk_count == 1
    assert result.index_dir == index_dir
    assert result.embedding_model == "text-embedding-v4"
    assert result.message == "FAISS 向量索引构建完成：1 个 chunk，模型 text-embedding-v4。"
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "chunks.json").exists()


def test_build_vector_index_from_chunks_skips_when_embedding_provider_missing(tmp_path):
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            {
                "chunk_id": "chunk-1",
                "content": "客户经理被投诉一次会影响评聘。",
                "source_file": "policy-a.pdf",
                "page_number": 2,
                "metadata": {},
            }
        ],
    )

    result = build_vector_index_from_chunks(
        chunks_path=chunks_path,
        index_dir=tmp_path / "runtime" / "vector_index",
        embedding_provider=None,
        embedding_model="text-embedding-v4",
    )

    assert result.status == VectorIndexBuildStatus.SKIPPED
    assert result.chunk_count == 0
    assert result.index_dir is None
    assert result.message == "未配置 Embedding Provider，已跳过 FAISS 向量索引构建。"


def test_build_vector_index_from_chunks_reports_missing_chunks_file(tmp_path):
    result = build_vector_index_from_chunks(
        chunks_path=tmp_path / "runtime" / "processed" / "chunks.jsonl",
        index_dir=tmp_path / "runtime" / "vector_index",
        embedding_provider=FakeEmbeddingProvider(),
        embedding_model="text-embedding-v4",
    )

    assert result.status == VectorIndexBuildStatus.MISSING_CHUNKS
    assert result.chunk_count == 0
    assert result.index_dir is None
    assert result.message == "未找到 chunks.jsonl，无法构建 FAISS 向量索引。"


def test_build_vector_index_from_chunks_reports_empty_chunks_file(tmp_path):
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text("", encoding="utf-8")

    result = build_vector_index_from_chunks(
        chunks_path=chunks_path,
        index_dir=tmp_path / "runtime" / "vector_index",
        embedding_provider=FakeEmbeddingProvider(),
        embedding_model="text-embedding-v4",
    )

    assert result.status == VectorIndexBuildStatus.EMPTY_CHUNKS
    assert result.chunk_count == 0
    assert result.index_dir is None
    assert result.message == "chunks.jsonl 为空，无法构建 FAISS 向量索引。"