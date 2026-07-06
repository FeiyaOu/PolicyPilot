from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.ingestion.chunk_store import read_chunks_jsonl


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_query(self, text: str) -> list[float]:
        pass


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    score: float
    content: str
    source_file: str
    page_number: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FaissVectorIndex:
    index: faiss.Index
    chunk_records: list[dict[str, Any]]

    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        top_k: int = 4,
    ) -> list[VectorSearchResult]:
        if not self.chunk_records or top_k <= 0:
            return []

        query_vector = _to_float32_matrix([embedding_provider.embed_query(query)])
        distances, indexes = self.index.search(query_vector, min(top_k, len(self.chunk_records)))
        scored_indexes = [
            (int(index), float(distance))
            for index, distance in zip(indexes[0], distances[0])
            if int(index) >= 0
        ]
        scores_by_index = _distances_to_scores(scored_indexes)

        return [
            self._to_result(index, scores_by_index[index])
            for index, _distance in scored_indexes
        ]

    def _to_result(self, index: int, score: float) -> VectorSearchResult:
        record = self.chunk_records[index]
        return VectorSearchResult(
            chunk_id=record["chunk_id"],
            score=score,
            content=record["content"],
            source_file=record["source_file"],
            page_number=record["page_number"],
            metadata=record.get("metadata", {}),
        )


def build_faiss_vector_index(
    chunk_records: list[dict[str, Any]],
    embedding_provider: EmbeddingProvider,
) -> FaissVectorIndex:
    index = faiss.IndexFlatL2(embedding_provider.dimension)
    if chunk_records:
        vectors = _to_float32_matrix(
            embedding_provider.embed_documents([record["content"] for record in chunk_records])
        )
        index.add(vectors)

    return FaissVectorIndex(index=index, chunk_records=chunk_records)


def build_faiss_vector_index_from_jsonl(
    chunks_path: str | Path,
    embedding_provider: EmbeddingProvider,
) -> FaissVectorIndex:
    return build_faiss_vector_index(read_chunks_jsonl(chunks_path), embedding_provider)


def save_faiss_vector_index(vector_index: FaissVectorIndex, index_dir: str | Path) -> Path:
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(vector_index.index, str(path / "index.faiss"))
    (path / "chunks.json").write_text(
        json.dumps(vector_index.chunk_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_faiss_vector_index(index_dir: str | Path) -> FaissVectorIndex:
    path = Path(index_dir)
    index = faiss.read_index(str(path / "index.faiss"))
    chunk_records = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    return FaissVectorIndex(index=index, chunk_records=chunk_records)


def _to_float32_matrix(vectors: list[list[float]]) -> np.ndarray:
    return np.asarray(vectors, dtype="float32")


def _distances_to_scores(indexed_distances: list[tuple[int, float]]) -> dict[int, float]:
    if not indexed_distances:
        return {}

    distances = [distance for _index, distance in indexed_distances]
    min_distance = min(distances)
    max_distance = max(distances)

    if min_distance == max_distance:
        return {index: 1.0 for index, _distance in indexed_distances}

    return {
        index: 1 - ((distance - min_distance) / (max_distance - min_distance))
        for index, distance in indexed_distances
    }
