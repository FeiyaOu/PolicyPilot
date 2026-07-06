from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from src.retrieval.bm25 import Bm25SearchResult
from src.retrieval.hybrid import fuse_hybrid_scores
from src.retrieval.vector_index import EmbeddingProvider, VectorSearchResult


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    content: str
    source_file: str
    page_number: int
    metadata: dict[str, Any]
    vector_score: float
    bm25_score: float
    fused_score: float


class VectorRetrieverLike(Protocol):
    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        top_k: int = 4,
    ) -> list[VectorSearchResult]:
        pass


class Bm25RetrieverLike(Protocol):
    def search(self, query: str, top_k: int = 4) -> list[Bm25SearchResult]:
        pass


@dataclass(frozen=True)
class RetrievalService:
    vector_retriever: VectorRetrieverLike | None
    bm25_retriever: Bm25RetrieverLike | None
    embedding_provider: EmbeddingProvider | None

    def search(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 4,
        alpha: float = 0.5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        if mode == RetrievalMode.VECTOR:
            return self._search_vector(query, top_k)
        if mode == RetrievalMode.BM25:
            return self._search_bm25(query, top_k)
        if mode == RetrievalMode.HYBRID:
            return self._search_hybrid(query, top_k, alpha)

        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def _search_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.vector_retriever is None or self.embedding_provider is None:
            return []

        return [
            _from_vector_result(result)
            for result in self.vector_retriever.search(query, self.embedding_provider, top_k=top_k)
        ]

    def _search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.bm25_retriever is None:
            return []

        return [_from_bm25_result(result) for result in self.bm25_retriever.search(query, top_k=top_k)]

    def _search_hybrid(self, query: str, top_k: int, alpha: float) -> list[RetrievalResult]:
        vector_results = self._search_vector(query, top_k)
        bm25_results = self._search_bm25(query, top_k)
        records_by_chunk_id = _records_by_chunk_id(vector_results, bm25_results)

        fused_scores = fuse_hybrid_scores(
            vector_scores={result.chunk_id: result.vector_score for result in vector_results},
            bm25_scores={result.chunk_id: result.bm25_score for result in bm25_results},
            alpha=alpha,
        )

        return [
            _with_scores(
                records_by_chunk_id[result.chunk_id],
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                fused_score=result.fused_score,
            )
            for result in fused_scores[:top_k]
            if result.chunk_id in records_by_chunk_id
        ]


def _from_vector_result(result: VectorSearchResult) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=result.chunk_id,
        content=result.content,
        source_file=result.source_file,
        page_number=result.page_number,
        metadata=result.metadata,
        vector_score=result.score,
        bm25_score=0.0,
        fused_score=result.score,
    )


def _from_bm25_result(result: Bm25SearchResult) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=result.chunk_id,
        content=result.content,
        source_file=result.source_file,
        page_number=result.page_number,
        metadata=result.metadata,
        vector_score=0.0,
        bm25_score=result.score,
        fused_score=result.score,
    )


def _records_by_chunk_id(*result_groups: list[RetrievalResult]) -> dict[str, RetrievalResult]:
    records: dict[str, RetrievalResult] = {}
    for results in result_groups:
        for result in results:
            records.setdefault(result.chunk_id, result)
    return records


def _with_scores(
    result: RetrievalResult,
    vector_score: float,
    bm25_score: float,
    fused_score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=result.chunk_id,
        content=result.content,
        source_file=result.source_file,
        page_number=result.page_number,
        metadata=result.metadata,
        vector_score=vector_score,
        bm25_score=bm25_score,
        fused_score=fused_score,
    )
