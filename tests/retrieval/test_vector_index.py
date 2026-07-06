from src.ingestion.models import DocumentChunk
from src.ingestion.chunk_store import write_chunks_jsonl
from src.retrieval.vector_index import (
    FaissVectorIndex,
    VectorSearchResult,
    build_faiss_vector_index,
    build_faiss_vector_index_from_jsonl,
    load_faiss_vector_index,
    save_faiss_vector_index,
)


class DeterministicEmbeddingProvider:
    dimension = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        return [
            1.0 if "投诉" in text else 0.0,
            1.0 if "评聘" in text else 0.0,
            1.0 if "营业" in text else 0.0,
        ]


def make_record(content: str, source_file: str, page_number: int, chunk_index: int) -> dict:
    return DocumentChunk(
        content=content,
        source_file=source_file,
        page_number=page_number,
        metadata={"chunk_index": chunk_index},
    ).to_dict()


def test_build_faiss_vector_index_searches_with_deterministic_embeddings():
    records = [
        make_record("客户经理被投诉一次会影响评聘。", "policy-a.pdf", 2, 0),
        make_record("网点营业时间调整需要提前公告。", "policy-b.pdf", 5, 0),
    ]
    embedding_provider = DeterministicEmbeddingProvider()

    vector_index = build_faiss_vector_index(records, embedding_provider)
    results = vector_index.search("投诉是否影响评聘？", embedding_provider, top_k=1)

    assert isinstance(vector_index, FaissVectorIndex)
    assert results == [
        VectorSearchResult(
            chunk_id=records[0]["chunk_id"],
            score=1.0,
            content=records[0]["content"],
            source_file="policy-a.pdf",
            page_number=2,
            metadata={"chunk_index": 0},
        )
    ]


def test_faiss_vector_index_persists_and_loads_searchable_index(tmp_path):
    records = [
        make_record("客户经理被投诉一次会影响评聘。", "policy-a.pdf", 2, 0),
        make_record("网点营业时间调整需要提前公告。", "policy-b.pdf", 5, 0),
    ]
    embedding_provider = DeterministicEmbeddingProvider()
    vector_index = build_faiss_vector_index(records, embedding_provider)
    index_dir = tmp_path / "runtime" / "indexes" / "faiss"

    save_faiss_vector_index(vector_index, index_dir)
    loaded_index = load_faiss_vector_index(index_dir)

    results = loaded_index.search("营业时间", embedding_provider, top_k=1)
    assert [result.chunk_id for result in results] == [records[1]["chunk_id"]]


def test_build_faiss_vector_index_handles_empty_records():
    vector_index = build_faiss_vector_index([], DeterministicEmbeddingProvider())

    assert vector_index.search("投诉", DeterministicEmbeddingProvider()) == []


def test_build_faiss_vector_index_from_jsonl_loads_processed_chunks(tmp_path):
    chunks = [
        DocumentChunk(
            content="客户经理被投诉一次会影响评聘。",
            source_file="policy-a.pdf",
            page_number=2,
            metadata={"chunk_index": 0},
        )
    ]
    chunks_path = tmp_path / "runtime" / "processed" / "chunks.jsonl"
    write_chunks_jsonl(chunks, chunks_path)

    vector_index = build_faiss_vector_index_from_jsonl(chunks_path, DeterministicEmbeddingProvider())

    results = vector_index.search("投诉评聘", DeterministicEmbeddingProvider(), top_k=1)
    assert [result.chunk_id for result in results] == [chunks[0].chunk_id]
