from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.app_services.build_service import build_knowledge_base
from src.app_services.upload_flow import UploadedFileLike, prepare_build_request


@dataclass(frozen=True)
class KnowledgeBaseBuildPageResult:
    document_count: int
    chunk_count: int
    warning_count: int
    output_path: Path | None
    message: str


def build_knowledge_base_from_ui(
    uploaded_files: list[UploadedFileLike] | None,
    raw_data_dir: str | Path,
    chunk_output_path: str | Path,
) -> KnowledgeBaseBuildPageResult:
    request = prepare_build_request(uploaded_files=uploaded_files, raw_data_dir=raw_data_dir)
    if not request.documents:
        return KnowledgeBaseBuildPageResult(
            document_count=0,
            chunk_count=0,
            warning_count=0,
            output_path=None,
            message="未选择 PDF，也未在 data/raw 中找到 PDF。",
        )

    output_path = Path(chunk_output_path)
    result = build_knowledge_base(request, chunk_output_path=output_path)
    return KnowledgeBaseBuildPageResult(
        document_count=result.summary.document_count,
        chunk_count=result.summary.chunk_count,
        warning_count=result.summary.warning_count,
        output_path=output_path,
        message=(
            "知识库构建完成："
            f"{result.summary.document_count} 个文档，"
            f"{result.summary.chunk_count} 个 chunk，"
            f"{result.summary.warning_count} 个警告。"
        ),
    )