from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ingestion.models import DocumentChunk


class PdfPage(Protocol):
    def extract_text(self) -> str | None:
        pass


class PdfReaderLike(Protocol):
    pages: list[PdfPage]


@dataclass(frozen=True)
class IngestionWarning:
    source_file: str
    page_number: int
    message: str


@dataclass(frozen=True)
class IngestionSummary:
    document_count: int
    chunk_count: int
    warning_count: int


@dataclass(frozen=True)
class PdfIngestionResult:
    chunks: list[DocumentChunk]
    warnings: list[IngestionWarning]
    summary: IngestionSummary


def ingest_pdf_reader(reader: PdfReaderLike, source_file: str) -> PdfIngestionResult:
    chunks: list[DocumentChunk] = []
    warnings: list[IngestionWarning] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            warnings.append(
                IngestionWarning(
                    source_file=source_file,
                    page_number=page_index,
                    message="No extractable text found on page",
                )
            )
            continue

        chunks.append(
            DocumentChunk(
                content=text.strip(),
                source_file=source_file,
                page_number=page_index,
            )
        )

    summary = IngestionSummary(
        document_count=1,
        chunk_count=len(chunks),
        warning_count=len(warnings),
    )

    return PdfIngestionResult(chunks=chunks, warnings=warnings, summary=summary)