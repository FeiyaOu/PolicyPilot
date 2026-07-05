import json

import pytest

from src.ingestion.chunk_store import read_chunks_jsonl, write_chunks_jsonl
from src.ingestion.models import DocumentChunk


def test_write_chunks_jsonl_creates_parent_directory_and_preserves_chunk_fields(tmp_path):
    chunks = [
        DocumentChunk(
            content="客户经理应保留投诉处理记录。",
            source_file="policy.pdf",
            page_number=3,
            metadata={"chunk_index": 0, "start_char": 10, "end_char": 24},
        ),
        DocumentChunk(
            content="评聘申报材料应按时提交。",
            source_file="policy.pdf",
            page_number=4,
            metadata={"chunk_index": 1, "start_char": 25, "end_char": 39},
        ),
    ]
    output_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"

    write_chunks_jsonl(chunks, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first_record = json.loads(lines[0])
    assert first_record == chunks[0].to_dict()


def test_read_chunks_jsonl_returns_records_in_file_order(tmp_path):
    chunk = DocumentChunk(
        content="客户经理应保留投诉处理记录。",
        source_file="policy.pdf",
        page_number=3,
        metadata={"chunk_index": 0},
    )
    output_path = tmp_path / "chunks.jsonl"
    output_path.write_text(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")

    records = read_chunks_jsonl(output_path)

    assert records == [chunk.to_dict()]


def test_write_chunks_jsonl_rejects_empty_chunk_list(tmp_path):
    with pytest.raises(ValueError, match="chunks must not be empty"):
        write_chunks_jsonl([], tmp_path / "chunks.jsonl")
