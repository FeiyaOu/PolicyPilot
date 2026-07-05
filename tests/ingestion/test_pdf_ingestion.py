from src.ingestion.chunk_splitter import SplitConfig
from src.ingestion.pdf_ingestion import ingest_pdf_reader


class FakePage:
    def __init__(self, text: str | None):
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakePdfReader:
    def __init__(self, pages: list[FakePage]):
        self.pages = pages


def test_ingest_pdf_reader_returns_chunks_with_source_metadata():
    reader = FakePdfReader(
        [
            FakePage("第一页政策内容：客户经理投诉处理规则。"),
            FakePage("第二页政策内容：年度评聘申报时间。"),
        ]
    )

    result = ingest_pdf_reader(reader, source_file="客户经理考核办法.pdf")

    assert result.summary.document_count == 1
    assert result.summary.chunk_count == 2
    assert result.summary.warning_count == 0
    assert [chunk.page_number for chunk in result.chunks] == [1, 2]
    assert {chunk.source_file for chunk in result.chunks} == {"客户经理考核办法.pdf"}
    assert all(chunk.chunk_id for chunk in result.chunks)


def test_ingest_pdf_reader_warns_for_pages_without_text_and_continues():
    reader = FakePdfReader(
        [
            FakePage("第一页有效政策内容。"),
            FakePage(None),
            FakePage("   "),
            FakePage("第四页有效政策内容。"),
        ]
    )

    result = ingest_pdf_reader(reader, source_file="policy.pdf")

    assert result.summary.document_count == 1
    assert result.summary.chunk_count == 2
    assert result.summary.warning_count == 2
    assert [chunk.page_number for chunk in result.chunks] == [1, 4]
    assert [warning.page_number for warning in result.warnings] == [2, 3]
    assert all("No extractable text" in warning.message for warning in result.warnings)


def test_ingest_pdf_reader_splits_long_page_text_with_page_metadata():
    reader = FakePdfReader(
        [
            FakePage("第一条客户经理应遵守投诉处理规则。第二条客户经理应按时完成评聘申报。第三条客户经理应保留处理记录。"),
        ]
    )

    result = ingest_pdf_reader(
        reader,
        source_file="policy.pdf",
        split_config=SplitConfig(chunk_size=28, chunk_overlap=8),
    )

    assert len(result.chunks) > 1
    assert result.summary.chunk_count == len(result.chunks)
    assert {chunk.page_number for chunk in result.chunks} == {1}
    assert [chunk.metadata["chunk_index"] for chunk in result.chunks] == list(range(len(result.chunks)))