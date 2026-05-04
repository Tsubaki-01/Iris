from collections.abc import Generator

import pytest
from pydantic import BaseModel, ValidationError

import iris
from iris.core import ModelRoute, create_provider_client, parse_model_route
from iris.exceptions import IrisConfigError, IrisProviderError, IrisValidationError
from iris.message import LLMRequest, Msg
from iris.providers import ProviderClient


@pytest.fixture(autouse=True)
def isolate_factory_config(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """隔离 factory 测试使用的环境变量与全局配置。"""
    for name in (
        "IRIS_OPENAI_API_KEY",
        "IRIS_ANTHROPIC_API_KEY",
        "IRIS_UNKNOWN_API_KEY",
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
    ],
)
def test_create_provider_client_selects_adapter(model: str, provider: str) -> None:
    client = create_provider_client(model, api_key="test-key")

    assert isinstance(client, ProviderClient)
    assert client.adapter.provider == provider
    assert client.api_key == "test-key"


def test_create_provider_client_rejects_unknown_provider() -> None:
    with pytest.raises(IrisProviderError):
        create_provider_client("unknown/model", api_key="test-key")


def test_create_provider_client_prefers_explicit_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_OPENAI_API_KEY", "env-key")

    client = create_provider_client("openai/gpt-4o", api_key="explicit-key")

    assert client.api_key == "explicit-key"


def test_create_provider_client_reads_provider_specific_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_OPENAI_API_KEY", "provider-key")

    client = create_provider_client("openai/gpt-4o")

    assert client.api_key == "provider-key"


def test_create_provider_client_falls_back_to_initialized_config() -> None:
    iris.init_config(api_key="generic-key")

    client = create_provider_client("openai/gpt-4o")

    assert client.api_key == "generic-key"


def test_create_provider_client_requires_api_key() -> None:
    with pytest.raises(IrisConfigError):
        create_provider_client("openai/gpt-4o")


def test_create_provider_client_accepts_model_route_for_request_model() -> None:
    route = parse_model_route("openai/gpt-4o")

    client = create_provider_client(route, api_key="test-key")
    request = LLMRequest(model=route.model, messages=[Msg.user("你好")])

    assert client.adapter.provider == "openai"
    assert request.model == "gpt-4o"
