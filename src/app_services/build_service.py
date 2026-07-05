from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PyPDF2 import PdfReader

from src.app_services.upload_flow import BuildRequest, DocumentInput
from src.ingestion.models import DocumentChunk
from src.ingestion.pdf_ingestion import (
    IngestionSummary,
    IngestionWarning,
    ingest_pdf_file,
    ingest_pdf_reader,
)


@dataclass(frozen=True)
class BuildResult:
    chunks: list[DocumentChunk]
    warnings: list[IngestionWarning]
    summary: IngestionSummary


def build_knowledge_base(request: BuildRequest) -> BuildResult:
    if not request.documents:
        raise ValueError("Build request must include at least one document")

    chunks: list[DocumentChunk] = []
    warnings: list[IngestionWarning] = []

    for document in request.documents:
        ingestion_result = _ingest_document(document)
        chunks.extend(ingestion_result.chunks)
        warnings.extend(ingestion_result.warnings)

    summary = IngestionSummary(
        document_count=len(request.documents),
        chunk_count=len(chunks),
        warning_count=len(warnings),
    )
    return BuildResult(chunks=chunks, warnings=warnings, summary=summary)


def _ingest_document(document: DocumentInput):
    if document.origin == "raw":
        if document.path is None:
            raise ValueError(f"Raw document requires a path: {document.source_file}")
        return ingest_pdf_file(document.path)

    if document.origin == "upload":
        if document.content is None:
            raise ValueError(f"Uploaded document requires content: {document.source_file}")
        reader = PdfReader(BytesIO(document.content))
        return ingest_pdf_reader(reader, source_file=document.source_file)

    raise ValueError(f"Unsupported document origin: {document.origin}")
