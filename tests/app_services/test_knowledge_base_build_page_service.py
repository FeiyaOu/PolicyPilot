from dataclasses import dataclass

from src.app_services.knowledge_base_build_page_service import build_knowledge_base_from_ui
from src.app_services.upload_flow import BuildRequest, DocumentInput
from src.ingestion.models import DocumentChunk
from src.ingestion.pdf_ingestion import IngestionSummary


@dataclass(frozen=True)
class FakeUploadedFile:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def test_build_knowledge_base_from_ui_persists_chunks_and_returns_summary(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    output_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    captured_request = None

    def fake_build_knowledge_base(request: BuildRequest, chunk_output_path):
        nonlocal captured_request
        captured_request = request
        chunk = DocumentChunk(content="客户经理被投诉一次会影响评聘。", source_file="policy-a.pdf", page_number=2)
        chunk_output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_output_path.write_text(str(chunk.to_dict()), encoding="utf-8")
        return type(
            "Result",
            (),
            {
                "chunks": [chunk],
                "warnings": [],
                "summary": IngestionSummary(document_count=1, chunk_count=1, warning_count=0),
            },
        )()

    monkeypatch.setattr(
        "src.app_services.knowledge_base_build_page_service.build_knowledge_base",
        fake_build_knowledge_base,
    )

    result = build_knowledge_base_from_ui(
        uploaded_files=[FakeUploadedFile("policy-a.pdf", b"pdf bytes")],
        raw_data_dir=raw_dir,
        chunk_output_path=output_path,
    )

    assert captured_request == BuildRequest(
        documents=[
            DocumentInput(
                source_file="policy-a.pdf",
                content=b"pdf bytes",
                path=None,
                origin="upload",
            )
        ]
    )
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.warning_count == 0
    assert result.output_path == output_path
    assert result.message == "知识库构建完成：1 个文档，1 个 chunk，0 个警告。"


def test_build_knowledge_base_from_ui_reports_empty_input_without_writing(tmp_path):
    result = build_knowledge_base_from_ui(
        uploaded_files=[],
        raw_data_dir=tmp_path / "missing_raw",
        chunk_output_path=tmp_path / "runtime" / "processed" / "chunks.jsonl",
    )

    assert result.document_count == 0
    assert result.chunk_count == 0
    assert result.warning_count == 0
    assert result.output_path is None
    assert result.message == "未选择 PDF，也未在 data/raw 中找到 PDF。"