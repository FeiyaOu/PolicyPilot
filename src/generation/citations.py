from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.app_services.retrieval_service import RetrievalResult


@dataclass(frozen=True)
class Citation:
    source_file: str
    page_number: int | None
    chunk_ids: tuple[str, ...]
    metadata: dict[str, Any]


def build_citations(retrieval_results: list[RetrievalResult]) -> list[Citation]:
    grouped: dict[tuple[str, int | None], dict[str, Any]] = {}

    for result in retrieval_results:
        source_file = result.source_file.strip() or "unknown source"
        page_number = result.page_number if result.page_number > 0 else None
        key = (source_file, page_number)

        if key not in grouped:
            grouped[key] = {"chunk_ids": [], "chunk_indexes": []}

        grouped[key]["chunk_ids"].append(result.chunk_id)
        if "chunk_index" in result.metadata:
            grouped[key]["chunk_indexes"].append(result.metadata["chunk_index"])

    return [
        Citation(
            source_file=source_file,
            page_number=page_number,
            chunk_ids=tuple(group["chunk_ids"]),
            metadata={"chunk_indexes": tuple(group["chunk_indexes"])},
        )
        for (source_file, page_number), group in grouped.items()
    ]
