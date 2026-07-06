from src.app_services.local_env import load_env_file


def test_load_env_file_loads_key_values_without_overriding_existing_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local config",
                "DASHSCOPE_API_KEY=local-secret",
                "POLICYPILOT_LLM_MODEL=qwen-plus",
                "POLICYPILOT_EMBEDDING_MODEL='text-embedding-v4'",
            ]
        ),
        encoding="utf-8",
    )
    env = {"DASHSCOPE_API_KEY": "shell-secret"}

    loaded = load_env_file(env_path, env=env)

    assert loaded is True
    assert env["DASHSCOPE_API_KEY"] == "shell-secret"
    assert env["POLICYPILOT_LLM_MODEL"] == "qwen-plus"
    assert env["POLICYPILOT_EMBEDDING_MODEL"] == "text-embedding-v4"


def test_load_env_file_returns_false_when_file_is_missing(tmp_path):
    env = {}

    loaded = load_env_file(tmp_path / ".env", env=env)

    assert loaded is False
    assert env == {}