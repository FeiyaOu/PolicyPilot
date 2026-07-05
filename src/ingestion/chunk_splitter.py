from __future__ import annotations

from dataclasses import dataclass, field

from src.ingestion.models import DocumentChunk


DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "，", ".", ";", ",", " ", ""]


@dataclass(frozen=True)
class SplitConfig:
    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: list[str] = field(default_factory=lambda: DEFAULT_SEPARATORS.copy())

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


def split_page_text(
    text: str,
    source_file: str,
    page_number: int,
    config: SplitConfig | None = None,
) -> list[DocumentChunk]:
    normalized_text = text.strip()
    if not normalized_text:
        return []

    split_config = config or SplitConfig()
    chunks: list[DocumentChunk] = []
    start_char = 0

    while start_char < len(normalized_text):
        end_char = _find_chunk_end(normalized_text, start_char, split_config)
        chunk_content = normalized_text[start_char:end_char].strip()

        if chunk_content:
            chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    source_file=source_file,
                    page_number=page_number,
                    metadata={
                        "chunk_index": len(chunks),
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                )
            )

        if end_char >= len(normalized_text):
            break

        start_char = max(0, end_char - split_config.chunk_overlap)

    return chunks


def _find_chunk_end(text: str, start_char: int, config: SplitConfig) -> int:
    hard_end = min(start_char + config.chunk_size, len(text))
    if hard_end >= len(text):
        return len(text)

    window = text[start_char:hard_end]
    for separator in config.separators:
        if not separator:
            continue
        separator_index = window.rfind(separator)
        if separator_index > 0:
            return start_char + separator_index + len(separator)

    return hard_end
