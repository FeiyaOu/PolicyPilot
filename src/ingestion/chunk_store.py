from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ingestion.models import DocumentChunk


def write_chunks_jsonl(chunks: list[DocumentChunk], output_path: str | Path) -> Path:
    if not chunks:
        raise ValueError("chunks must not be empty")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    return path


def read_chunks_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records
