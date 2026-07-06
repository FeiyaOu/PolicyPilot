from src.app_services.retrieval_service import RetrievalMode, RetrievalResult, RetrievalService
from src.retrieval.bm25 import Bm25SearchResult
from src.retrieval.vector_index import VectorSearchResult


class FakeVectorRetriever:
    def search(self, query, embedding_provider, top_k=4):
        return [
            VectorSearchResult(
                chunk_id="chunk-1",
                score=0.8,
                content="客户经理被投诉一次会影响评聘。",
                source_file="policy-a.pdf",
                page_number=2,
                metadata={"chunk_index": 0},
            ),
            VectorSearchResult(
                chunk_id="chunk-2",
                score=0.3,
                content="网点营业时间调整需要提前公告。",
                source_file="policy-b.pdf",
                page_number=5,
                metadata={"chunk_index": 0},
            ),
        ][:top_k]


class FakeBm25Retriever:
    def search(self, query, top_k=4):
        return [
            Bm25SearchResult(
                chunk_id="chunk-1",
                score=0.4,
                content="客户经理被投诉一次会影响评聘。",
                source_file="policy-a.pdf",
                page_number=2,
                metadata={"chunk_index": 0},
            ),
            Bm25SearchResult(
                chunk_id="chunk-3",
                score=0.7,
                content="投诉记录需要保留处理材料。",
                source_file="policy-c.pdf",
                page_number=8,
                metadata={"chunk_index": 1},
            ),
        ][:top_k]


class FakeEmbeddingProvider:
    dimension = 3

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _text in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_retrieval_service_returns_vector_only_results_with_source_metadata():
    service = RetrievalService(
        vector_retriever=FakeVectorRetriever(),
        bm25_retriever=FakeBm25Retriever(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search("投诉是否影响评聘？", mode=RetrievalMode.VECTOR, top_k=1)

    assert results == [
        RetrievalResult(
            chunk_id="chunk-1",
            content="客户经理被投诉一次会影响评聘。",
            source_file="policy-a.pdf",
            page_number=2,
            metadata={"chunk_index": 0},
            vector_score=0.8,
            bm25_score=0.0,
            fused_score=0.8,
        )
    ]


def test_retrieval_service_returns_bm25_only_results_with_source_metadata():
    service = RetrievalService(
        vector_retriever=FakeVectorRetriever(),
        bm25_retriever=FakeBm25Retriever(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search("投诉是否影响评聘？", mode=RetrievalMode.BM25, top_k=1)

    assert results == [
        RetrievalResult(
            chunk_id="chunk-1",
            content="客户经理被投诉一次会影响评聘。",
            source_file="policy-a.pdf",
            page_number=2,
            metadata={"chunk_index": 0},
            vector_score=0.0,
            bm25_score=0.4,
            fused_score=0.4,
        )
    ]


def test_retrieval_service_returns_hybrid_results_ranked_by_fused_score():
    service = RetrievalService(
        vector_retriever=FakeVectorRetriever(),
        bm25_retriever=FakeBm25Retriever(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search("投诉是否影响评聘？", mode=RetrievalMode.HYBRID, top_k=3, alpha=0.5)

    assert [(result.chunk_id, result.vector_score, result.bm25_score, result.fused_score) for result in results] == [
        ("chunk-1", 0.8, 0.4, 0.6000000000000001),
        ("chunk-3", 0.0, 0.7, 0.35),
        ("chunk-2", 0.3, 0.0, 0.15),
    ]
    assert results[0].source_file == "policy-a.pdf"
    assert results[0].page_number == 2


def test_retrieval_service_handles_empty_results():
    service = RetrievalService(
        vector_retriever=None,
        bm25_retriever=None,
        embedding_provider=None,
    )

    assert service.search("投诉是否影响评聘？", mode=RetrievalMode.HYBRID) == []
