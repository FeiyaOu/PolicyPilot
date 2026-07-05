import pytest

from src.ingestion.models import DocumentChunk
from src.ingestion.chunk_store import write_chunks_jsonl
from src.retrieval.bm25 import Bm25Retriever, build_bm25_retriever_from_jsonl, tokenize_chinese_text


def make_record(content: str, source_file: str, page_number: int, chunk_index: int) -> dict:
    return DocumentChunk(
        content=content,
        source_file=source_file,
        page_number=page_number,
        metadata={"chunk_index": chunk_index},
    ).to_dict()


def test_tokenize_chinese_text_returns_non_empty_terms():
    tokens = tokenize_chinese_text("客户经理投诉扣分规则")

    assert "客户经理" in tokens
    assert "投诉" in tokens
    assert "扣分" in tokens


def test_bm25_retriever_returns_normalized_results_with_metadata():
    records = [
        make_record("客户经理被投诉一次会扣分，并影响年度评聘。", "policy-a.pdf", 2, 0),
        make_record("客户经理每年需要按时提交评聘申报材料。", "policy-b.pdf", 5, 0),
        make_record("网点营业时间调整需要提前公告。", "policy-c.pdf", 1, 0),
    ]
    retriever = Bm25Retriever(records)

    results = retriever.search("投诉扣分", top_k=2)

    assert [result.chunk_id for result in results] == [records[0]["chunk_id"]]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].content == records[0]["content"]
    assert results[0].source_file == "policy-a.pdf"
    assert results[0].page_number == 2
    assert results[0].metadata == {"chunk_index": 0}


def test_bm25_retriever_returns_empty_list_for_no_match_or_empty_corpus():
    retriever = Bm25Retriever([
        make_record("客户经理每年需要按时提交评聘申报材料。", "policy-b.pdf", 5, 0),
    ])
    empty_retriever = Bm25Retriever([])

    assert retriever.search("不存在的流程", top_k=3) == []
    assert empty_retriever.search("投诉扣分", top_k=3) == []


def test_build_bm25_retriever_from_jsonl_loads_chunk_records(tmp_path):
    chunks = [
        DocumentChunk(
            content="客户经理被投诉一次会扣分。",
            source_file="policy-a.pdf",
            page_number=2,
            metadata={"chunk_index": 0},
        )
    ]
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    write_chunks_jsonl(chunks, chunks_path)

    retriever = build_bm25_retriever_from_jsonl(chunks_path)

    results = retriever.search("投诉扣分")
    assert [result.chunk_id for result in results] == [chunks[0].chunk_id]
