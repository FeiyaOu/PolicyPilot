from http import HTTPStatus

from src.app_services.llm_provider_config import LlmProviderStatus, build_llm_provider_from_env
from src.generation.answer_generator import AnswerGenerationInput


class FakeMessage:
    content = "会影响，需依据证据中的投诉性质和处理结果判断。"


class FakeChoice:
    message = FakeMessage()


class FakeOutput:
    choices = [FakeChoice()]


class FakeResponse:
    status_code = HTTPStatus.OK
    output = FakeOutput()


class FakeGenerationClient:
    calls = []

    @classmethod
    def call(cls, **kwargs):
        cls.calls.append(kwargs)
        return FakeResponse()


def test_build_llm_provider_returns_missing_when_dashscope_key_is_absent():
    result = build_llm_provider_from_env(env={})

    assert result.status == LlmProviderStatus.MISSING
    assert result.provider is None
    assert result.provider_name == "dashscope"
    assert result.api_key_configured is False
    assert result.message == "未配置 DashScope API Key，将使用本地演示回答。"


def test_build_llm_provider_returns_dashscope_provider_when_key_is_present():
    result = build_llm_provider_from_env(
        env={"DASHSCOPE_API_KEY": "secret-key", "POLICYPILOT_LLM_MODEL": "qwen-plus"},
        generation_client=FakeGenerationClient,
    )

    assert result.status == LlmProviderStatus.READY
    assert result.provider is not None
    assert result.provider_name == "dashscope"
    assert result.model == "qwen-plus"
    assert result.api_key_configured is True
    assert "secret-key" not in repr(result.provider)


def test_dashscope_llm_provider_generates_answer_from_grounded_prompt():
    FakeGenerationClient.calls = []
    result = build_llm_provider_from_env(
        env={"DASHSCOPE_API_KEY": "secret-key"},
        generation_client=FakeGenerationClient,
    )

    answer = result.provider.generate(
        AnswerGenerationInput(
            question="客户经理投诉会影响评聘吗？",
            contexts=[
                {
                    "chunk_id": "chunk-1",
                    "content": "客户经理被投诉一次会影响评聘。",
                    "source_file": "policy-a.pdf",
                    "page_number": 2,
                    "score": 0.6,
                }
            ],
        )
    )

    assert answer == "会影响，需依据证据中的投诉性质和处理结果判断。"
    assert FakeGenerationClient.calls[0]["model"] == "qwen-plus"
    assert FakeGenerationClient.calls[0]["api_key"] == "secret-key"
    assert FakeGenerationClient.calls[0]["result_format"] == "message"
    assert FakeGenerationClient.calls[0]["messages"][0]["role"] == "system"
    assert "客户经理被投诉一次会影响评聘。" in FakeGenerationClient.calls[0]["messages"][1]["content"]


def test_build_llm_provider_rejects_unsupported_provider():
    result = build_llm_provider_from_env(
        env={"POLICYPILOT_LLM_PROVIDER": "other", "DASHSCOPE_API_KEY": "secret-key"},
        generation_client=FakeGenerationClient,
    )

    assert result.status == LlmProviderStatus.INVALID
    assert result.provider is None
    assert result.message == "不支持的 LLM Provider: other"