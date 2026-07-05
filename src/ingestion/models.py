from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    source_file: str
    page_number: int
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("content must not be empty")
        if self.page_number <= 0:
            raise ValueError("page_number must be a positive integer")

        object.__setattr__(self, "chunk_id", self._generate_chunk_id())

    def _generate_chunk_id(self) -> str:
        raw_id = f"{self.source_file}:{self.page_number}:{self.content.strip()}"
        return sha256(raw_id.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }