import json
from pathlib import Path

from src.app_services.knowledge_base_loader import KnowledgeBaseStatus, load_knowledge_base


def write_chunks(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "chunk_id": "chunk-1",
            "content": "客户经理被投诉一次会影响评聘。",
            "source_file": "policy-a.pdf",
            "page_number": 2,
            "metadata": {"chunk_index": 0},
        }
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")


def test_load_knowledge_base_returns_missing_status_when_chunks_file_does_not_exist(tmp_path):
    result = load_knowledge_base(tmp_path / "runtime" / "processed" / "chunks.jsonl")

    assert result.status == KnowledgeBaseStatus.MISSING
    assert result.chunk_count == 0
    assert result.retrieval_service is None
    assert result.message == "未找到知识库 chunks.jsonl，请先构建或加载知识库。"


def test_load_knowledge_base_builds_bm25_retrieval_service_from_chunks_jsonl(tmp_path):
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    write_chunks(chunks_path)

    result = load_knowledge_base(chunks_path)

    assert result.status == KnowledgeBaseStatus.READY
    assert result.chunk_count == 1
    assert result.retrieval_service is not None
    assert result.message == "已加载 1 个知识库 chunk。"

    search_results = result.retrieval_service.search("投诉 评聘", top_k=1)

    assert search_results[0].chunk_id == "chunk-1"
    assert search_results[0].source_file == "policy-a.pdf"
    assert search_results[0].page_number == 2


def test_load_knowledge_base_returns_empty_status_for_empty_chunks_file(tmp_path):
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text("", encoding="utf-8")

    result = load_knowledge_base(chunks_path)

    assert result.status == KnowledgeBaseStatus.EMPTY
    assert result.chunk_count == 0
    assert result.retrieval_service is None
    assert result.message == "知识库 chunks.jsonl 为空，请重新构建知识库。"