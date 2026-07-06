from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
from typing import Any, Mapping


class EmbeddingProviderStatus(StrEnum):
    READY = "ready"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True)
class EmbeddingProviderConfigResult:
    status: EmbeddingProviderStatus
    provider: DashScopeEmbeddingProvider | None
    provider_name: str
    model: str
    dimension: int | None
    api_key_configured: bool
    message: str


@dataclass(frozen=True)
class DashScopeEmbeddingProvider:
    api_key: str = field(repr=False)
    model: str = "text-embedding-v4"
    dimension: int = 1024
    text_embedding_client: Any = field(repr=False, default=None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, text_type="query")[0]

    def _embed(self, input_value: str | list[str], text_type: str) -> list[list[float]]:
        client = self.text_embedding_client or _load_dashscope_text_embedding_client()
        response = client.call(
            model=self.model,
            input=input_value,
            api_key=self.api_key,
            text_type=text_type,
            dimension=self.dimension,
        )
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope embedding request failed: {response.status_code}")

        return [item.embedding for item in response.output.embeddings]


def build_embedding_provider_from_env(
    env: Mapping[str, str] | None = None,
    text_embedding_client: Any = None,
) -> EmbeddingProviderConfigResult:
    values = env if env is not None else os.environ
    provider_name = values.get("POLICYPILOT_EMBEDDING_PROVIDER", "dashscope").strip().lower()
    model = values.get("POLICYPILOT_EMBEDDING_MODEL", "text-embedding-v4")
    raw_dimension = values.get("POLICYPILOT_EMBEDDING_DIMENSION", "1024")

    try:
        dimension = int(raw_dimension)
    except ValueError:
        return EmbeddingProviderConfigResult(
            status=EmbeddingProviderStatus.INVALID,
            provider=None,
            provider_name=provider_name,
            model=model,
            dimension=None,
            api_key_configured=False,
            message="POLICYPILOT_EMBEDDING_DIMENSION 必须是整数。",
        )

    if provider_name != "dashscope":
        return EmbeddingProviderConfigResult(
            status=EmbeddingProviderStatus.INVALID,
            provider=None,
            provider_name=provider_name,
            model=model,
            dimension=dimension,
            api_key_configured=False,
            message=f"不支持的 Embedding Provider: {provider_name}",
        )

    api_key = values.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return EmbeddingProviderConfigResult(
            status=EmbeddingProviderStatus.MISSING,
            provider=None,
            provider_name=provider_name,
            model=model,
            dimension=dimension,
            api_key_configured=False,
            message="未配置 DashScope API Key，向量检索暂不可用。",
        )

    provider = DashScopeEmbeddingProvider(
        api_key=api_key,
        model=model,
        dimension=dimension,
        text_embedding_client=text_embedding_client,
    )
    return EmbeddingProviderConfigResult(
        status=EmbeddingProviderStatus.READY,
        provider=provider,
        provider_name=provider_name,
        model=model,
        dimension=dimension,
        api_key_configured=True,
        message=f"已配置 DashScope Embedding Provider: {model}。",
    )


def _load_dashscope_text_embedding_client():
    from dashscope import TextEmbedding

    return TextEmbedding