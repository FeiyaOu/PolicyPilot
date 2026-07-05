import pytest

from src.ingestion.models import DocumentChunk


def test_document_chunk_contains_required_metadata():
    chunk = DocumentChunk(
        content="客户经理被投诉一次，按照制度扣减相应考核分。",
        source_file="客户经理考核办法.pdf",
        page_number=3,
        metadata={"section": "投诉处理"},
    )

    assert chunk.chunk_id
    assert chunk.content == "客户经理被投诉一次，按照制度扣减相应考核分。"
    assert chunk.source_file == "客户经理考核办法.pdf"
    assert chunk.page_number == 3
    assert chunk.metadata == {"section": "投诉处理"}


def test_document_chunk_rejects_empty_content():
    with pytest.raises(ValueError, match="content"):
        DocumentChunk(content="  ", source_file="policy.pdf", page_number=1)


def test_document_chunk_requires_positive_page_number():
    with pytest.raises(ValueError, match="page_number"):
        DocumentChunk(content="有效内容", source_file="policy.pdf", page_number=0)


def test_document_chunk_id_is_stable_for_same_source_page_and_content():
    first = DocumentChunk(content="同一段政策内容", source_file="policy.pdf", page_number=2)
    second = DocumentChunk(content="同一段政策内容", source_file="policy.pdf", page_number=2)

    assert first.chunk_id == second.chunk_id


def test_document_chunk_can_be_converted_to_dict():
    chunk = DocumentChunk(
        content="评聘申报时间为每年固定周期。",
        source_file="policy.pdf",
        page_number=5,
        metadata={"version": "v1"},
    )

    assert chunk.to_dict() == {
        "chunk_id": chunk.chunk_id,
        "content": "评聘申报时间为每年固定周期。",
        "source_file": "policy.pdf",
        "page_number": 5,
        "metadata": {"version": "v1"},
    }