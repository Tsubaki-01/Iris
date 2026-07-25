from collections.abc import Generator

import pytest
from pydantic import BaseModel, ValidationError

import iris
from iris.config import ProviderConfig
from iris.exceptions import IrisConfigError, IrisProviderError, IrisValidationError
from iris.message import LLMRequest, Msg
from iris.providers import (
    ModelRoute,
    ProviderClient,
    create_provider_client,
    parse_model_route,
)


@pytest.fixture(autouse=True)
def isolate_provider_factory_config(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """隔离 provider factory 测试使用的环境变量与全局配置。"""
    for name in (
        "IRIS_PROVIDER_API_KEYS__OPENAI",
        "IRIS_PROVIDER_API_KEYS__ANTHROPIC",
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK",
        "IRIS_PROVIDER_API_KEYS__UNKNOWN",
        "IRIS_PROVIDER_API_KEYS__SILICONFLOW",
    ):
        monkeypatch.delenv(name, raising=False)
    iris.reset()
    yield
    iris.reset()


def test_model_route_is_frozen_pydantic_model() -> None:
    route = ModelRoute(provider="openai", model="gpt-4o")

    assert isinstance(route, BaseModel)
    with pytest.raises(ValidationError):
        route.model = "gpt-4o-mini"


def test_parse_model_route_strips_provider_prefix() -> None:
    route = parse_model_route("openai/gpt-4o")

    assert route == ModelRoute(provider="openai", model="gpt-4o")


def test_parse_model_route_splits_only_first_slash() -> None:
    route = parse_model_route("openai/gpt/4o")

    assert route == ModelRoute(provider="openai", model="gpt/4o")


@pytest.mark.parametrize("model", ["", "gpt-4o", "/gpt-4o", "openai/"])
def test_parse_model_route_rejects_invalid_model_strings(model: str) -> None:
    with pytest.raises(IrisValidationError):
        parse_model_route(model)


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-sonnet-4-5", "anthropic"),
        ("deepseek/deepseek-chat", "deepseek"),
    ],
)
def test_create_provider_client_selects_supported_provider(model: str, provider: str) -> None:
    client = create_provider_client(model, api_key="test-key")

    assert isinstance(client, ProviderClient)
    assert client.provider == provider
    assert client.api_key == "test-key"


def test_create_provider_client_rejects_unknown_provider() -> None:
    with pytest.raises(IrisProviderError):
        create_provider_client("unknown/model", api_key="test-key")


def test_create_provider_client_prefers_explicit_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_PROVIDER_API_KEYS__OPENAI", "env-key")
    iris.init_config(provider_api_keys={"openai": "config-key"})

    client = create_provider_client("openai/gpt-4o", api_key="explicit-key")

    assert client.api_key == "explicit-key"


def test_create_provider_client_reads_provider_specific_key_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_PROVIDER_API_KEYS__OPENAI", "provider-key")
    iris.init_config()

    client = create_provider_client("openai/gpt-4o")

    assert client.api_key == "provider-key"


def test_create_provider_client_does_not_read_os_environ_after_config_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iris.init_config(
        api_key="generic-key",
        provider_api_keys={"deepseek": "config-provider-key"},
    )
    monkeypatch.setenv("IRIS_PROVIDER_API_KEYS__DEEPSEEK", "env-key")

    client = create_provider_client("deepseek/deepseek-chat")

    assert client.provider == "deepseek"
    assert client.api_key == "config-provider-key"


def test_create_provider_client_falls_back_to_initialized_config() -> None:
    iris.init_config(api_key="generic-key")

    client = create_provider_client("deepseek/deepseek-chat")

    assert client.provider == "deepseek"
    assert client.api_key == "generic-key"


def test_create_provider_client_allows_explicit_key_without_config_secret() -> None:
    iris.init_config(
        providers={
            "siliconflow": ProviderConfig(
                litellm_provider="openai",
                base_url="https://api.siliconflow.cn/v1",
            )
        }
    )

    client = create_provider_client(
        "siliconflow/deepseek-ai/DeepSeek-V3",
        api_key="explicit-key",
    )

    assert client.api_key == "explicit-key"


def test_create_provider_client_preserves_builtin_litellm_provider_on_override() -> None:
    iris.init_config(
        provider_api_keys={"deepseek": "deepseek-key"},
        providers={"deepseek": ProviderConfig(base_url="https://api.deepseek.example/v1")},
    )

    client = create_provider_client("deepseek/deepseek-chat")

    assert client.litellm_provider == "deepseek"
    assert client.base_url == "https://api.deepseek.example/v1"


def test_create_provider_client_ignores_blank_generic_api_key() -> None:
    iris.init_config(api_key="   ")

    with pytest.raises(IrisConfigError):
        create_provider_client("openai/gpt-4o")


def test_create_provider_client_uses_registered_custom_provider() -> None:
    iris.init_config(
        provider_api_keys={"siliconflow": "siliconflow-key"},
        providers={
            "siliconflow": ProviderConfig(
                litellm_provider="openai",
                base_url="https://api.siliconflow.cn/v1",
            )
        },
    )

    client = create_provider_client("siliconflow/deepseek-ai/DeepSeek-V3")

    assert client.provider == "siliconflow"
    assert client.litellm_provider == "openai"
    assert client.api_key == "siliconflow-key"
    assert client.base_url == "https://api.siliconflow.cn/v1"


def test_create_provider_client_requires_custom_provider_base_url() -> None:
    iris.init_config(provider_api_keys={"siliconflow": "siliconflow-key"})

    with pytest.raises(IrisProviderError):
        create_provider_client("siliconflow/deepseek-ai/DeepSeek-V3")


def test_create_provider_client_prefers_explicit_base_url() -> None:
    iris.init_config(
        provider_api_keys={"siliconflow": "siliconflow-key"},
        providers={
            "siliconflow": ProviderConfig(
                litellm_provider="openai",
                base_url="https://api.siliconflow.cn/v1",
            )
        },
    )

    client = create_provider_client(
        "siliconflow/deepseek-ai/DeepSeek-V3",
        base_url="https://proxy.example.test/v1",
    )

    assert client.base_url == "https://proxy.example.test/v1"


def test_create_provider_client_requires_api_key() -> None:
    with pytest.raises(IrisConfigError):
        create_provider_client("openai/gpt-4o")


def test_create_provider_client_accepts_model_route_for_request_model() -> None:
    route = parse_model_route("openai/gpt-4o")

    client = create_provider_client(route, api_key="test-key")
    request = LLMRequest(model=route.model, messages=[Msg.user("你好")])

    assert client.provider == "openai"
    assert request.model == "gpt-4o"


def test_create_provider_client_rejects_removed_http_client_keyword() -> None:
    with pytest.raises(TypeError):
        create_provider_client("openai/gpt-4o", api_key="test-key", http_client=None)
