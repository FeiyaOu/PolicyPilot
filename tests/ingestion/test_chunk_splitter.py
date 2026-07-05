from src.ingestion.chunk_splitter import SplitConfig, split_page_text


def test_short_page_text_returns_single_chunk_with_metadata():
    chunks = split_page_text(
        text="客户经理投诉处理规则。",
        source_file="policy.pdf",
        page_number=2,
        config=SplitConfig(chunk_size=100, chunk_overlap=20),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "客户经理投诉处理规则。"
    assert chunks[0].source_file == "policy.pdf"
    assert chunks[0].page_number == 2
    assert chunks[0].metadata == {
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len("客户经理投诉处理规则。"),
    }


def test_long_page_text_is_split_into_multiple_chunks_with_overlap():
    text = "第一条客户经理应遵守投诉处理规则。第二条客户经理应按时完成评聘申报。第三条客户经理应保留处理记录。"

    chunks = split_page_text(
        text=text,
        source_file="policy.pdf",
        page_number=1,
        config=SplitConfig(chunk_size=28, chunk_overlap=8),
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 28 for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[1].metadata["start_char"] < chunks[0].metadata["end_char"]
    assert chunks[1].metadata["start_char"] == chunks[0].metadata["end_char"] - 8


def test_splitter_prefers_chinese_sentence_boundaries():
    text = "客户经理应及时响应投诉。客户经理应记录处理过程。客户经理应完成后续回访。"

    chunks = split_page_text(
        text=text,
        source_file="policy.pdf",
        page_number=1,
        config=SplitConfig(chunk_size=18, chunk_overlap=0),
    )

    assert chunks[0].content.endswith("。")
    assert chunks[1].content.endswith("。")


def test_empty_page_text_returns_no_chunks():
    assert split_page_text(text="   ", source_file="policy.pdf", page_number=1) == []
