import json
from typing import Any

import pytest
from litellm import ModelResponse

from iris.exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from iris.message import LLMRequest, Msg, ToolUseBlock
from iris.providers import ProviderClient


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
async def test_provider_client_parses_current_litellm_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_response = ModelResponse(
        id="chatcmpl-current",
        model="gpt-4o",
        choices=[{"message": {"content": "当前契约"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        del kwargs
        return raw_response

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)

    response = await ProviderClient(provider="openai", api_key="test-key").complete(
        LLMRequest(model="gpt-4o")
    )

    assert response.id == "chatcmpl-current"
    assert response.to_msg().text == "当前契约"
    assert response.total_tokens == 3


@pytest.mark.asyncio
async def test_provider_client_uses_litellm_provider_for_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    seen_model = ""

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        nonlocal seen_model
        seen_model = str(kwargs["model"])
        return {"choices": [{"message": {"content": "你好"}}]}

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    client = ProviderClient(
        provider="siliconflow",
        litellm_provider="openai",
        api_key="test-key",
    )

    await client.complete(LLMRequest(model="deepseek-ai/DeepSeek-V3", messages=[Msg.user("你好")]))

    assert seen_model == "openai/deepseek-ai/DeepSeek-V3"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_provider_rejects_non_chat_request_before_network_io(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """原始 provider 选项不能把当前 Chat 调用静默改成不支持的 API。"""
    import iris.providers.client as provider_client

    async def unexpected_call(**kwargs: Any) -> None:
        pytest.fail("unsupported API style must not reach LiteLLM")

    monkeypatch.setattr(provider_client.litellm, "acompletion", unexpected_call)
    client = ProviderClient(provider="openai", api_key="test-key")
    request = LLMRequest(model="gpt-4o", stream=stream, provider_options={"api_style": "responses"})
    with pytest.raises(IrisProviderError, match="不支持的 provider API 风格"):
        if stream:
            await anext(client.stream(request))
        else:
            await client.complete(request)


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    [
        ("gpt-4o", "openai", "gpt-4o"),
        ("openai/gpt-4o", "openai", "gpt-4o"),
        ("anthropic/claude-3.5-sonnet", "openrouter", "anthropic/claude-3.5-sonnet"),
        ("openrouter/anthropic/claude-3.5-sonnet", "openrouter", "anthropic/claude-3.5-sonnet"),
    ],
)
def test_input_estimate_uses_complete_wire_messages_and_preserves_inner_route(
    monkeypatch: pytest.MonkeyPatch, model: str, provider: str, expected: str
) -> None:
    import iris.providers.client as provider_client

    calls: list[dict[str, Any]] = []

    def count(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 10 if "messages" in kwargs else 3

    async def unexpected_generation(**kwargs: Any) -> None:
        pytest.fail("input estimation must not generate a response")

    monkeypatch.setattr(provider_client.litellm, "token_counter", count)
    monkeypatch.setattr(provider_client.litellm, "acompletion", unexpected_generation)
    request = LLMRequest(
        model=model,
        messages=[
            Msg.system("规则"),
            Msg.user("<summary>旧摘要</summary>", sender="context"),
            Msg.assistant(content=[ToolUseBlock(id="c1", name="lookup", input={"q": "值"})]),
            Msg.tool_result(tool_use_id="c1", content="结果正文"),
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="auto",
        response_format={"type": "json_schema", "json_schema": {"name": "答案"}},
        timeout=123,
    )
    client = ProviderClient(provider="gateway", litellm_provider=provider, api_key="test")

    assert client.estimate_input_tokens(request) == 13
    assert calls[0] == {
        "model": expected,
        "messages": client._to_litellm_kwargs(request)["messages"],
        "tools": request.tools,
        "tool_choice": "auto",
    }
    assert calls[0]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "结果正文"}
    assert len(calls) == 2
    assert calls[1]["model"] == expected
    assert json.loads(calls[1]["text"]) == request.response_format


def test_input_estimate_without_response_schema_counts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    calls: list[dict[str, Any]] = []

    def count(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 7

    monkeypatch.setattr(provider_client.litellm, "token_counter", count)
    client = ProviderClient(provider="openai", api_key="test")
    assert client.estimate_input_tokens(LLMRequest(model="gpt-4o")) == 7
    assert len(calls) == 1


def test_input_estimate_maps_counter_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import iris.providers.client as provider_client

    def count(**kwargs: Any) -> int:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(provider_client.litellm, "token_counter", count)
    with pytest.raises(IrisProviderError, match="tokenizer unavailable"):
        ProviderClient(provider="openai", api_key="test").estimate_input_tokens(
            LLMRequest(model="gpt-4o")
        )


def test_request_retry_override_preserves_zero_without_changing_default() -> None:
    client = ProviderClient(provider="openai", api_key="test")
    assert "num_retries" not in client._to_litellm_kwargs(LLMRequest(model="gpt-4o"))
    assert (
        client._to_litellm_kwargs(LLMRequest(model="gpt-4o", provider_options={"num_retries": 0}))[
            "num_retries"
        ]
        == 0
    )
