from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes:
        pass


@dataclass(frozen=True)
class DocumentInput:
    source_file: str
    content: bytes | None
    path: Path | None
    origin: str


@dataclass(frozen=True)
class BuildRequest:
    documents: list[DocumentInput]


def prepare_uploaded_pdf(uploaded_file: UploadedFileLike) -> DocumentInput:
    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError(f"Only PDF files are supported: {uploaded_file.name}")

    return DocumentInput(
        source_file=uploaded_file.name,
        content=uploaded_file.getvalue(),
        path=None,
        origin="upload",
    )


def prepare_build_request(
    uploaded_files: list[UploadedFileLike] | None,
    raw_data_dir: str | Path,
) -> BuildRequest:
    documents: list[DocumentInput] = []
    raw_dir = Path(raw_data_dir)

    if raw_dir.exists():
        for pdf_path in sorted(raw_dir.glob("*.pdf")):
            documents.append(
                DocumentInput(
                    source_file=pdf_path.name,
                    content=None,
                    path=pdf_path,
                    origin="raw",
                )
            )

    for uploaded_file in uploaded_files or []:
        documents.append(prepare_uploaded_pdf(uploaded_file))

    return BuildRequest(documents=documents)
