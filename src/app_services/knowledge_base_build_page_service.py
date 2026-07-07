from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.app_services.build_service import build_knowledge_base
from src.app_services.upload_flow import UploadedFileLike, prepare_build_request
from src.ingestion.chunk_store import read_chunks_jsonl


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

    # Preserve manually-added chunks (e.g. from conversation extraction)
    # so they are not wiped when the PDF knowledge base is rebuilt.
    preserved: list[dict] = []
    if output_path.exists():
        preserved = [
            c for c in read_chunks_jsonl(output_path)
            if c.get("source_file") == "对话知识沉淀"
        ]

    result = build_knowledge_base(request, chunk_output_path=output_path)

    # Re-append preserved chunks that are not already present
    if preserved:
        import json as _json
        existing = read_chunks_jsonl(output_path)
        existing_ids = {c["chunk_id"] for c in existing}
        to_add = [c for c in preserved if c["chunk_id"] not in existing_ids]
        if to_add:
            with output_path.open("a", encoding="utf-8") as f:
                for c in to_add:
                    f.write(_json.dumps(c, ensure_ascii=False) + "\n")

    total_chunks = result.summary.chunk_count + len(preserved)
    preserved_label = f"（含 {len(preserved)} 条手动知识）" if preserved else ""
    return KnowledgeBaseBuildPageResult(
        document_count=result.summary.document_count,
        chunk_count=total_chunks,
        warning_count=result.summary.warning_count,
        output_path=output_path,
        message=(
            f"知识库构建完成："
            f"{result.summary.document_count} 个文档，"
            f"{total_chunks} 个 chunk{preserved_label}，"
            f"{result.summary.warning_count} 个警告。"
        ),
    )