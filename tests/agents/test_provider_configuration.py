"""Provider 构造时的协议路由、显式覆盖与凭据归属。"""

from pathlib import Path

import pytest

from iris.config import Config, ProviderConfig
from iris.providers import create_provider_client


@pytest.mark.parametrize(
    ("api_style", "transport"),
    [("responses", "openai"), ("chat_completions", "deepseek")],
)
def test_deepseek_protocol_defaults_and_logical_credentials(
    monkeypatch: pytest.MonkeyPatch, api_style: str, transport: str
) -> None:
    import iris.providers.factory as factory

    config = Config(
        provider_api_keys={"deepseek": "deepseek-key", "openai": "other-key"},
        providers={"deepseek": ProviderConfig(headers={"x-config": "yes"})},
    )
    monkeypatch.setattr(factory, "is_config_initialized", lambda: True)
    monkeypatch.setattr(factory, "get_config", lambda: config)
    client = create_provider_client("deepseek/model", api_style=api_style)
    assert client.api_style == api_style
    assert client.litellm_provider == transport
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_key == "deepseek-key"
    assert client.headers == {"x-config": "yes"}


def test_explicit_transport_and_endpoint_override_protocol_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.factory as factory

    config = Config(
        providers={
            "deepseek": ProviderConfig(litellm_provider="openai", base_url="https://config.test/v1")
        }
    )
    monkeypatch.setattr(factory, "is_config_initialized", lambda: True)
    monkeypatch.setattr(factory, "get_config", lambda: config)
    client = create_provider_client(
        "deepseek/model",
        api_key="explicit",
        api_style="chat_completions",
        base_url="https://explicit.test/v1",
    )
    assert client.litellm_provider == "openai"
    assert client.base_url == "https://explicit.test/v1"
    assert client.api_key == "explicit"


def test_factory_defaults_to_responses() -> None:
    assert create_provider_client("openai/model", api_key="test").api_style == "responses"


@pytest.mark.parametrize("api_style", ["responses", "chat_completions"])
def test_runtime_assembly_binds_model_protocol_without_request_override(
    tmp_path: Path, api_style: str
) -> None:
    from iris.agents import AgentConfig
    from iris.runtime import RuntimeFactory

    config = AgentConfig.model_validate(
        {
            "name": "protocol-test",
            "model": {"provider": "openai", "name": "model", "api_style": api_style},
            "system": "instructions",
            "permissions": {"workspace": str(tmp_path)},
            "context_policy": {"enabled": False},
        }
    )
    runtime = RuntimeFactory.from_config(config, api_key="test")
    assert runtime.environment.provider.api_style == api_style
    assert "api_style" not in config.model.to_llm_request_options()
