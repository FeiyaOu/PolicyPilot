from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion import pdf_ingestion


class FakePage:
    def __init__(self, text: str | None):
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakePdf:
    def __init__(self, *args, **kwargs):
        self.pages = [FakePage("测试 PDF 第一页政策内容。")]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_ingest_pdf_file_loads_pdf_reader_and_preserves_file_name(tmp_path, monkeypatch):
    import pdfplumber
    pdf_path = tmp_path / "客户经理考核办法.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake test pdf")
    monkeypatch.setattr(pdfplumber, "open", FakePdf)

    result = pdf_ingestion.ingest_pdf_file(pdf_path)

    assert result.summary.document_count == 1
    assert result.summary.chunk_count == 1
    assert result.chunks[0].source_file == "客户经理考核办法.pdf"
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].content == "测试 PDF 第一页政策内容。"


def test_ingest_pdf_file_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="missing.pdf"):
        pdf_ingestion.ingest_pdf_file(missing_path)


def test_ingest_pdf_file_rejects_non_pdf_file(tmp_path):
    text_path = tmp_path / "policy.txt"
    text_path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="PDF"):
        pdf_ingestion.ingest_pdf_file(text_path)