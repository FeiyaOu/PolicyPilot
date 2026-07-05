from src.app_services.upload_flow import BuildRequest, DocumentInput
from src.ingestion.models import DocumentChunk
from src.ingestion.pdf_ingestion import IngestionSummary, IngestionWarning, PdfIngestionResult
from src.app_services import build_service


class FakePdfReader:
    def __init__(self, stream):
        self.stream = stream


def make_result(source_file: str, page_number: int, content: str, warning_count: int = 0) -> PdfIngestionResult:
    warnings = [
        IngestionWarning(
            source_file=source_file,
            page_number=page_number + 1,
            message="No extractable text found on page",
        )
        for _ in range(warning_count)
    ]
    chunks = [
        DocumentChunk(
            content=content,
            source_file=source_file,
            page_number=page_number,
        )
    ]
    return PdfIngestionResult(
        chunks=chunks,
        warnings=warnings,
        summary=IngestionSummary(
            document_count=1,
            chunk_count=len(chunks),
            warning_count=len(warnings),
        ),
    )


def test_build_knowledge_base_ingests_raw_file_documents(tmp_path, monkeypatch):
    raw_pdf = tmp_path / "默认政策.pdf"
    raw_pdf.write_bytes(b"raw pdf")
    request = BuildRequest(
        documents=[
            DocumentInput(source_file="默认政策.pdf", content=None, path=raw_pdf, origin="raw")
        ]
    )
    called_paths = []

    def fake_ingest_pdf_file(path):
        called_paths.append(path)
        return make_result("默认政策.pdf", 1, "默认政策内容")

    monkeypatch.setattr(build_service, "ingest_pdf_file", fake_ingest_pdf_file)

    result = build_service.build_knowledge_base(request)

    assert called_paths == [raw_pdf]
    assert result.summary.document_count == 1
    assert result.summary.chunk_count == 1
    assert result.chunks[0].source_file == "默认政策.pdf"


def test_build_knowledge_base_ingests_uploaded_documents_from_bytes(monkeypatch):
    request = BuildRequest(
        documents=[
            DocumentInput(
                source_file="上传政策.pdf",
                content=b"uploaded pdf bytes",
                path=None,
                origin="upload",
            )
        ]
    )
    captured_source_files = []

    def fake_ingest_pdf_reader(reader, source_file):
        captured_source_files.append(source_file)
        assert isinstance(reader, FakePdfReader)
        assert reader.stream.getvalue() == b"uploaded pdf bytes"
        return make_result(source_file, 1, "上传政策内容")

    monkeypatch.setattr(build_service, "PdfReader", FakePdfReader)
    monkeypatch.setattr(build_service, "ingest_pdf_reader", fake_ingest_pdf_reader)

    result = build_service.build_knowledge_base(request)

    assert captured_source_files == ["上传政策.pdf"]
    assert result.summary.document_count == 1
    assert result.chunks[0].content == "上传政策内容"


def test_build_knowledge_base_merges_chunks_warnings_and_summary(tmp_path, monkeypatch):
    raw_pdf = tmp_path / "默认政策.pdf"
    raw_pdf.write_bytes(b"raw pdf")
    request = BuildRequest(
        documents=[
            DocumentInput(source_file="默认政策.pdf", content=None, path=raw_pdf, origin="raw"),
            DocumentInput(source_file="上传政策.pdf", content=b"uploaded pdf", path=None, origin="upload"),
        ]
    )

    def fake_ingest_pdf_file(path):
        return make_result("默认政策.pdf", 1, "默认政策内容", warning_count=1)

    def fake_ingest_pdf_reader(reader, source_file):
        return make_result(source_file, 2, "上传政策内容", warning_count=2)

    monkeypatch.setattr(build_service, "PdfReader", FakePdfReader)
    monkeypatch.setattr(build_service, "ingest_pdf_file", fake_ingest_pdf_file)
    monkeypatch.setattr(build_service, "ingest_pdf_reader", fake_ingest_pdf_reader)

    result = build_service.build_knowledge_base(request)

    assert [chunk.source_file for chunk in result.chunks] == ["默认政策.pdf", "上传政策.pdf"]
    assert result.summary.document_count == 2
    assert result.summary.chunk_count == 2
    assert result.summary.warning_count == 3
    assert len(result.warnings) == 3


def test_build_knowledge_base_rejects_empty_build_request():
    request = BuildRequest(documents=[])

    try:
        build_service.build_knowledge_base(request)
    except ValueError as error:
        assert "document" in str(error)
    else:
        raise AssertionError("Expected empty build request to raise ValueError")
