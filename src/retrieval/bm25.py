from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from src.ingestion.chunk_store import read_chunks_jsonl


@dataclass(frozen=True)
class Bm25SearchResult:
    chunk_id: str
    score: float
    content: str
    source_file: str
    page_number: int
    metadata: dict[str, Any]


class Bm25Retriever:
    def __init__(self, chunk_records: list[dict[str, Any]]):
        self.chunk_records = chunk_records
        self.tokenized_corpus = [tokenize_chinese_text(record["content"]) for record in chunk_records]
        self._bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(self, query: str, top_k: int = 4) -> list[Bm25SearchResult]:
        if self._bm25 is None or top_k <= 0:
            return []

        query_tokens = tokenize_chinese_text(query)
        if not query_tokens:
            return []

        raw_scores = list(self._bm25.get_scores(query_tokens))
        matched_indexes = [
            index
            for index, document_tokens in enumerate(self.tokenized_corpus)
            if set(query_tokens) & set(document_tokens)
        ]
        if not matched_indexes:
            return []

        normalized_scores = _normalize_scores({index: raw_scores[index] for index in matched_indexes})
        results = [
            self._to_result(index, normalized_scores[index])
            for index in matched_indexes
            if normalized_scores[index] > 0
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def _to_result(self, index: int, score: float) -> Bm25SearchResult:
        record = self.chunk_records[index]
        return Bm25SearchResult(
            chunk_id=record["chunk_id"],
            score=score,
            content=record["content"],
            source_file=record["source_file"],
            page_number=record["page_number"],
            metadata=record.get("metadata", {}),
        )


def tokenize_chinese_text(text: str) -> list[str]:
    return [token.strip() for token in jieba.lcut(text.lower()) if token.strip()]


def build_bm25_retriever_from_jsonl(chunks_path: str | Path) -> Bm25Retriever:
    return Bm25Retriever(read_chunks_jsonl(chunks_path))


def _normalize_scores(scores_by_index: dict[int, float]) -> dict[int, float]:
    if not scores_by_index:
        return {}

    max_score = max(scores_by_index.values())
    min_score = min(scores_by_index.values())

    if max_score == min_score:
        return {index: 1.0 for index in scores_by_index}

    return {
        index: (score - min_score) / (max_score - min_score)
        for index, score in scores_by_index.items()
    }
