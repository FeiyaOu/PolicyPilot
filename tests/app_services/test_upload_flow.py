from pathlib import Path

import pytest

from src.app_services.upload_flow import prepare_build_request, prepare_uploaded_pdf


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_uploaded_pdf_is_converted_to_document_input():
    uploaded = FakeUploadedFile("客户经理考核办法.pdf", b"fake pdf bytes")

    document = prepare_uploaded_pdf(uploaded)

    assert document.source_file == "客户经理考核办法.pdf"
    assert document.content == b"fake pdf bytes"
    assert document.origin == "upload"


def test_build_request_merges_uploaded_pdfs_and_raw_directory_pdfs(tmp_path):
    raw_pdf = tmp_path / "默认政策.pdf"
    raw_pdf.write_bytes(b"raw pdf bytes")
    ignored_text = tmp_path / "说明.txt"
    ignored_text.write_text("not a pdf", encoding="utf-8")
    uploaded = FakeUploadedFile("上传政策.pdf", b"uploaded pdf bytes")

    request = prepare_build_request(uploaded_files=[uploaded], raw_data_dir=tmp_path)

    assert [document.source_file for document in request.documents] == ["默认政策.pdf", "上传政策.pdf"]
    assert [document.origin for document in request.documents] == ["raw", "upload"]


def test_uploaded_non_pdf_is_rejected_with_user_readable_error():
    uploaded = FakeUploadedFile("policy.txt", b"not a pdf")

    with pytest.raises(ValueError, match="PDF"):
        prepare_uploaded_pdf(uploaded)


def test_build_request_contains_inputs_only_not_ingested_chunks(tmp_path):
    uploaded = FakeUploadedFile("policy.pdf", b"uploaded pdf bytes")

    request = prepare_build_request(uploaded_files=[uploaded], raw_data_dir=tmp_path)

    assert request.documents[0].source_file == "policy.pdf"
    assert not hasattr(request, "chunks")
    assert not hasattr(request, "index_path")
