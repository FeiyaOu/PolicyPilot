from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PyPDF2 import PdfReader

from src.ingestion.chunk_splitter import SplitConfig, split_page_text
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


def ingest_pdf_reader(
    reader: PdfReaderLike,
    source_file: str,
    split_config: SplitConfig | None = None,
) -> PdfIngestionResult:
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

        chunks.extend(
            split_page_text(
                text=text,
                source_file=source_file,
                page_number=page_index,
                config=split_config,
            )
        )

    summary = IngestionSummary(
        document_count=1,
        chunk_count=len(chunks),
        warning_count=len(warnings),
    )

    return PdfIngestionResult(chunks=chunks, warnings=warnings, summary=summary)


def ingest_pdf_file(pdf_path: str | Path, split_config: SplitConfig | None = None) -> PdfIngestionResult:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")

    reader = PdfReader(path)
    return ingest_pdf_reader(reader, source_file=path.name, split_config=split_config)