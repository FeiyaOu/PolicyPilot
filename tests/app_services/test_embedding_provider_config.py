from http import HTTPStatus

from src.app_services.embedding_provider_config import (
    EmbeddingProviderStatus,
    build_embedding_provider_from_env,
)


class FakeEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeOutput:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeResponse:
    status_code = HTTPStatus.OK

    def __init__(self, embeddings):
        self.output = FakeOutput([FakeEmbedding(embedding) for embedding in embeddings])


class FakeTextEmbeddingClient:
    calls = []

    @classmethod
    def call(cls, **kwargs):
        cls.calls.append(kwargs)
        input_value = kwargs["input"]
        texts = input_value if isinstance(input_value, list) else [input_value]
        return FakeResponse([[1.0, 0.0] for _text in texts])


def test_build_embedding_provider_returns_missing_when_dashscope_key_is_absent():
    result = build_embedding_provider_from_env(env={})

    assert result.status == EmbeddingProviderStatus.MISSING
    assert result.provider is None
    assert result.provider_name == "dashscope"
    assert result.api_key_configured is False
    assert result.message == "未配置 DashScope API Key，向量检索暂不可用。"


def test_build_embedding_provider_returns_dashscope_provider_when_key_is_present():
    result = build_embedding_provider_from_env(
        env={
            "DASHSCOPE_API_KEY": "secret-key",
            "POLICYPILOT_EMBEDDING_MODEL": "text-embedding-v4",
            "POLICYPILOT_EMBEDDING_DIMENSION": "2",
        },
        text_embedding_client=FakeTextEmbeddingClient,
    )

    assert result.status == EmbeddingProviderStatus.READY
    assert result.provider is not None
    assert result.provider_name == "dashscope"
    assert result.model == "text-embedding-v4"
    assert result.dimension == 2
    assert result.api_key_configured is True
    assert "secret-key" not in repr(result.provider)


def test_dashscope_provider_embeds_documents_and_query_with_text_type():
    FakeTextEmbeddingClient.calls = []
    result = build_embedding_provider_from_env(
        env={"DASHSCOPE_API_KEY": "secret-key", "POLICYPILOT_EMBEDDING_DIMENSION": "2"},
        text_embedding_client=FakeTextEmbeddingClient,
    )

    assert result.provider.embed_documents(["制度一", "制度二"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert result.provider.embed_query("制度问题") == [1.0, 0.0]
    assert [call["text_type"] for call in FakeTextEmbeddingClient.calls] == ["document", "query"]
    assert all(call["api_key"] == "secret-key" for call in FakeTextEmbeddingClient.calls)


def test_build_embedding_provider_rejects_invalid_dimension():
    result = build_embedding_provider_from_env(
        env={"DASHSCOPE_API_KEY": "secret-key", "POLICYPILOT_EMBEDDING_DIMENSION": "invalid"},
        text_embedding_client=FakeTextEmbeddingClient,
    )

    assert result.status == EmbeddingProviderStatus.INVALID
    assert result.provider is None
    assert result.message == "POLICYPILOT_EMBEDDING_DIMENSION 必须是整数。"