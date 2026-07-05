from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    score: float
    content: str


@dataclass(frozen=True)
class MultiQueryResult:
    original_query: str
    query_variants: tuple[str, ...]
    hits: list[RetrievalHit]


def normalize_query_variants(original_query: str, query_variants: list[str]) -> tuple[str, ...]:
    queries: list[str] = []

    for query in [original_query, *query_variants]:
        normalized_query = query.strip()
        if normalized_query and normalized_query not in queries:
            queries.append(normalized_query)

    return tuple(queries)


def merge_multi_query_results(
    original_query: str,
    query_variants: list[str],
    hits_by_query: dict[str, list[RetrievalHit]],
) -> MultiQueryResult:
    normalized_queries = normalize_query_variants(original_query, query_variants)
    best_hits_by_chunk_id: dict[str, RetrievalHit] = {}

    for query in normalized_queries:
        for hit in hits_by_query.get(query, []):
            current_hit = best_hits_by_chunk_id.get(hit.chunk_id)
            if current_hit is None or hit.score > current_hit.score:
                best_hits_by_chunk_id[hit.chunk_id] = hit

    hits = sorted(best_hits_by_chunk_id.values(), key=lambda hit: hit.score, reverse=True)
    return MultiQueryResult(original_query=original_query, query_variants=normalized_queries, hits=hits)
