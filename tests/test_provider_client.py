from typing import Any

import pytest
from pydantic import ValidationError

from iris.exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from iris.message import LLMRequest, Msg
from iris.providers import ProviderClient


def test_provider_client_exposes_litellm_active_fields_only() -> None:
    assert "max_retries" not in ProviderClient.model_fields
    assert "http_client" not in ProviderClient.model_fields
    assert "provider" in ProviderClient.model_fields


def test_provider_client_rejects_unsupported_provider() -> None:
    with pytest.raises(IrisProviderError):
        ProviderClient(provider="groq", api_key="test-key")


def test_provider_client_rejects_removed_http_client_keyword() -> None:
    with pytest.raises(ValidationError):
        ProviderClient(provider="openai", api_key="test-key", http_client=None)


def test_provider_client_rejects_removed_adapter_keyword() -> None:
    with pytest.raises(ValidationError):
        ProviderClient(provider="openai", api_key="test-key", adapter=object())


def test_provider_client_does_not_expose_adapter_or_close_compatibility() -> None:
    client = ProviderClient(provider="openai", api_key="test-key")

    assert not hasattr(client, "adapter")
    assert not hasattr(client, "close")


@pytest.mark.asyncio
async def test_provider_client_calls_litellm_with_openai_chat_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    seen_kwargs: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        seen_kwargs.update(kwargs)
        return {
            "id": "chatcmpl_1",
            "model": "gpt-4o",
            "object": "chat.completion",
            "choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    client = ProviderClient(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        timeout=30,
        headers={"x-trace-id": "trace-1"},
    )

    response = await client.complete(
        LLMRequest(
            model="gpt-4o",
            messages=[Msg.system("规则"), Msg.user("你好")],
            temperature=0,
            top_p=0,
            max_tokens=12,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice="auto",
            response_format={"type": "json_object"},
            timeout=5,
            provider_options={
                "api_style": "chat",
                "reasoning_effort": "low",
                "ignored_option": "ignored",
            },
        )
    )

    assert seen_kwargs == {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "你好", "name": "user"},
        ],
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "extra_headers": {"x-trace-id": "trace-1"},
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 12,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "timeout": 5,
        "reasoning_effort": "low",
    }
    assert response.provider == "openai"
    assert response.id == "chatcmpl_1"
    assert response.to_msg().text == "你好"
    assert response.input_tokens == 1
    assert response.output_tokens == 2
    assert response.total_tokens == 3


@pytest.mark.asyncio
async def test_provider_client_does_not_double_prefix_litellm_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    seen_model = ""

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        nonlocal seen_model
        seen_model = str(kwargs["model"])
        return {"choices": [{"message": {"content": "你好"}}]}

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    client = ProviderClient(provider="openai", api_key="test-key")

    await client.complete(
        LLMRequest(model="openai/gpt-4o", messages=[Msg.user("你好")])
    )

    assert seen_model == "openai/gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_style", ["responses", "unknown"])
async def test_provider_client_rejects_non_chat_api_style(api_style: str) -> None:
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(IrisProviderError, match=api_style):
        await client.complete(
            LLMRequest(
                model="gpt-4o",
                messages=[Msg.user("你好")],
                provider_options={"api_style": api_style},
            )
        )


class _FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("失败")
        self.status_code = status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, IrisAuthenticationError),
        (403, IrisAuthenticationError),
        (408, IrisAPIConnectionError),
        (429, IrisRateLimitExceededError),
        (500, IrisProviderError),
    ],
)
async def test_provider_client_maps_litellm_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_error: type[Exception],
) -> None:
    import iris.providers.client as provider_client

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        raise _FakeStatusError(status_code)

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(expected_error):
        await client.complete(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))


@pytest.mark.asyncio
async def test_provider_client_maps_litellm_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    class APIConnectionError(Exception):
        pass

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        raise APIConnectionError("无法连接")

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(IrisAPIConnectionError):
        await client.complete(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))


@pytest.mark.asyncio
async def test_provider_client_rejects_streaming_in_complete() -> None:
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(IrisProviderError, match="stream"):
        await client.complete(LLMRequest(model="gpt-4o", stream=True))
