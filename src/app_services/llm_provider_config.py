from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
from typing import Any, Mapping

from src.generation.answer_generator import AnswerGenerationInput, AnswerProvider
from src.generation.prompt_builder import build_grounded_answer_messages


class LlmProviderStatus(StrEnum):
    READY = "ready"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True)
class LlmProviderConfigResult:
    status: LlmProviderStatus
    provider: DashScopeLlmProvider | None
    provider_name: str
    model: str
    api_key_configured: bool
    message: str


@dataclass(frozen=True)
class DashScopeLlmProvider(AnswerProvider):
    api_key: str = field(repr=False)
    model: str = "qwen-plus"
    generation_client: Any = field(repr=False, default=None)

    def generate(self, generation_input: AnswerGenerationInput) -> str:
        client = self.generation_client or _load_dashscope_generation_client()
        response = client.call(
            model=self.model,
            messages=build_grounded_answer_messages(generation_input),
            result_format="message",
            api_key=self.api_key,
        )
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope generation request failed: {response.status_code}")

        return response.output.choices[0].message.content


def build_llm_provider_from_env(
    env: Mapping[str, str] | None = None,
    generation_client: Any = None,
) -> LlmProviderConfigResult:
    values = env if env is not None else os.environ
    provider_name = values.get("POLICYPILOT_LLM_PROVIDER", "dashscope").strip().lower()
    model = values.get("POLICYPILOT_LLM_MODEL", "qwen-plus")

    if provider_name != "dashscope":
        return LlmProviderConfigResult(
            status=LlmProviderStatus.INVALID,
            provider=None,
            provider_name=provider_name,
            model=model,
            api_key_configured=False,
            message=f"不支持的 LLM Provider: {provider_name}",
        )

    api_key = values.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return LlmProviderConfigResult(
            status=LlmProviderStatus.MISSING,
            provider=None,
            provider_name=provider_name,
            model=model,
            api_key_configured=False,
            message="未配置 DashScope API Key，将使用本地演示回答。",
        )

    provider = DashScopeLlmProvider(api_key=api_key, model=model, generation_client=generation_client)
    return LlmProviderConfigResult(
        status=LlmProviderStatus.READY,
        provider=provider,
        provider_name=provider_name,
        model=model,
        api_key_configured=True,
        message=f"已配置 DashScope LLM Provider: {model}。",
    )


def _load_dashscope_generation_client():
    from dashscope import Generation

    return Generation