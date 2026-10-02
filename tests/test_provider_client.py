import json
from typing import Any

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse

from iris.exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from iris.message import LLMRequest, Msg, ToolUseBlock
from iris.providers import ProviderClient
from iris.providers.responses import ResponsesMapper


def _response(text: str = "你好", **updates: Any) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "created_at": 0,
        "model": "gpt-4o",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        **updates,
    }


@pytest.mark.asyncio
async def test_provider_client_encodes_native_responses_adapter_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    seen_kwargs: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> dict[str, Any]:
        seen_kwargs.update(kwargs)
        return _response()

    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: fake_invoke(**kwargs)),
    )
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
            tools=[{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
            tool_choice="auto",
            response_format="json_object",
            timeout=5,
            provider_options={
                "reasoning_effort": "low",
                "ignored_option": "ignored",
            },
        )
    )

    assert seen_kwargs == {
        "model": "openai/gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "规则"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "你好"}],
            },
        ],
        "api_key": "test-key",
        "api_base": "https://example.test/v1",
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "extra_headers": {"x-trace-id": "trace-1"},
        "temperature": 0,
        "top_p": 0,
        "max_output_tokens": 12,
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "tool_choice": "auto",
        "text": {"format": {"type": "json_object"}},
        "timeout": 5,
        "reasoning": {"effort": "low"},
    }
    assert response.provider == "openai"
    assert response.id == "resp_1"
    assert response.to_msg().text == "你好"
    assert response.input_tokens == 1
    assert response.output_tokens == 2
    assert response.total_tokens == 3


@pytest.mark.asyncio
async def test_provider_client_parses_current_litellm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_response = ResponsesAPIResponse(**_response("当前契约", id="resp-current"))

    async def fake_invoke(**kwargs: Any) -> ResponsesAPIResponse:
        del kwargs
        return raw_response

    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: fake_invoke(**kwargs)),
    )

    response = await ProviderClient(provider="openai", api_key="test-key").complete(
        LLMRequest(model="gpt-4o")
    )

    assert response.id == "resp-current"
    assert response.to_msg().text == "当前契约"
    assert response.total_tokens == 3


@pytest.mark.asyncio
async def test_provider_client_uses_transport_provider_for_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    seen_model = ""

    async def fake_invoke(**kwargs: Any) -> dict[str, Any]:
        nonlocal seen_model
        seen_model = str(kwargs["model"])
        return _response()

    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: fake_invoke(**kwargs)),
    )
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
async def test_provider_client_maps_sdk_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_error: type[Exception],
) -> None:
    import iris.providers.client as provider_client

    async def fake_invoke(**kwargs: Any) -> dict[str, Any]:
        raise _FakeStatusError(status_code)

    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: fake_invoke(**kwargs)),
    )
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(expected_error):
        await client.complete(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))


@pytest.mark.asyncio
async def test_provider_client_maps_sdk_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    class APIConnectionError(Exception):
        pass

    async def fake_invoke(**kwargs: Any) -> dict[str, Any]:
        raise APIConnectionError("无法连接")

    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: fake_invoke(**kwargs)),
    )
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(IrisAPIConnectionError):
        await client.complete(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))


@pytest.mark.asyncio
async def test_provider_client_rejects_streaming_in_complete() -> None:
    client = ProviderClient(provider="openai", api_key="test-key")

    with pytest.raises(IrisProviderError, match="stream"):
        await client.complete(LLMRequest(model="gpt-4o", stream=True))


def test_api_style_is_owned_by_client_constructor() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProviderClient(provider="openai", api_key="test", api_style="chat")
    assert ProviderClient(provider="openai", api_key="test").api_style == "responses"


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    [
        ("gpt-4o", "openai", "gpt-4o"),
        ("openai/gpt-4o", "openai", "gpt-4o"),
        ("vendor/model", "openai", "vendor/model"),
        ("openai/vendor/model", "openai", "vendor/model"),
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
    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: unexpected_generation(**kwargs)),
    )
    request = LLMRequest(
        model=model,
        messages=[
            Msg.system("规则"),
            Msg.user("<summary>旧摘要</summary>", sender="context"),
            Msg.assistant(content=[ToolUseBlock(id="c1", name="lookup", input={"q": "值"})]),
            Msg.tool_result(tool_use_id="c1", content="结果正文"),
        ],
        tools=[{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
        tool_choice="auto",
        response_format={"name": "答案", "schema": {"type": "object"}},
        timeout=123,
    )
    client = ProviderClient(provider="gateway", litellm_provider=provider, api_key="test")

    assert client.estimate_input_tokens(request) == 13
    assert calls[0] == {
        "model": expected,
        "messages": ResponsesMapper().token_count_messages(
            client._to_adapter_kwargs(request)["input"]
        ),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                },
            }
        ],
        "tool_choice": "auto",
    }
    assert calls[0]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "结果正文"}
    assert len(calls) == 2
    assert calls[1]["model"] == expected
    assert json.loads(calls[1]["text"]) == {
        "type": "json_schema",
        "name": "答案",
        "schema": {"type": "object"},
    }


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
    assert "num_retries" not in client._to_adapter_kwargs(LLMRequest(model="gpt-4o"))
    assert (
        client._to_adapter_kwargs(LLMRequest(model="gpt-4o", provider_options={"num_retries": 0}))[
            "num_retries"
        ]
        == 0
    )


@pytest.mark.asyncio
async def test_chat_adapter_complete_and_plain_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    import iris.providers.client as provider_client
    from iris.runtime._compaction_summary import consume_summary_response

    captured: dict[str, Any] = {}

    async def chat(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": "chat-1",
            "model": "model",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "summary",
                        "reasoning_content": "thought",
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    async def no_responses(**kwargs: Any) -> None:
        pytest.fail("Chat must never change protocols")

    monkeypatch.setattr(provider_client.litellm, "acompletion", chat)
    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: no_responses(**kwargs)),
    )
    client = ProviderClient(
        provider="deepseek", api_key="test", api_style="chat_completions", timeout=40
    )
    result = await client.complete(
        LLMRequest(
            model="model",
            messages=[Msg.system("summarize"), Msg.user("history")],
            max_tokens=50,
            timeout=4,
            provider_options={"num_retries": 0, "reasoning_effort": "low"},
        )
    )
    assert consume_summary_response(result) == "summary"
    assert result.reasoning == "thought" and result.total_tokens == 5
    assert captured == {
        "model": "deepseek/model",
        "messages": [
            {"role": "system", "content": "summarize"},
            {"role": "user", "content": "history", "name": "user"},
        ],
        "api_key": "test",
        "max_tokens": 50,
        "timeout": 4,
        "num_retries": 0,
        "reasoning_effort": "low",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
async def test_chat_complete_incomplete_keeps_usage_without_tools(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str | None
) -> None:
    import iris.providers.client as provider_client

    async def chat(**kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "tool_calls": [
                            {"id": "call-1", "function": {"name": "lookup", "arguments": "{"}}
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr(provider_client.litellm, "acompletion", chat)
    client = ProviderClient(provider="openai", api_key="test", api_style="chat_completions")
    with pytest.raises(IrisProviderError) as exc_info:
        await client.complete(LLMRequest(model="gpt-4o"))
    assert exc_info.value.context["usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "[]", None])
async def test_chat_complete_invalid_tool_keeps_usage(
    monkeypatch: pytest.MonkeyPatch, arguments: str | None
) -> None:
    import iris.providers.client as provider_client

    async def chat(**kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {"id": "call-1", "function": {"name": "lookup", "arguments": arguments}}
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr(provider_client.litellm, "acompletion", chat)
    client = ProviderClient(provider="openai", api_key="test", api_style="chat_completions")
    with pytest.raises(IrisProviderError) as exc_info:
        await client.complete(LLMRequest(model="gpt-4o"))
    assert exc_info.value.context["usage"]["total_tokens"] == 5


@pytest.mark.parametrize("api_style", ["responses", "chat_completions"])
def test_protocol_token_estimate_covers_forced_tool_and_structured_format(
    monkeypatch: pytest.MonkeyPatch, api_style: str
) -> None:
    import iris.providers.client as provider_client
    from iris.message import ToolSpec

    calls: list[dict[str, Any]] = []

    def count(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 5

    monkeypatch.setattr(provider_client.litellm, "token_counter", count)
    client = ProviderClient(provider="openai", api_key="test", api_style=api_style)
    request = LLMRequest(
        model="gpt-4o",
        messages=[Msg.assistant("history")],
        tools=[ToolSpec(name="lookup", input_schema={"type": "object"})],
        tool_choice={"name": "lookup"},
        response_format={"name": "result", "schema": {"type": "object"}, "strict": True},
    )
    assert client.estimate_input_tokens(request) == 10
    assert calls[0]["messages"] == [{"role": "assistant", "content": "history"}]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert calls[0]["tools"][0]["function"]["parameters"] == {"type": "object"}
    expected_format = (
        {"type": "json_schema", **request.response_format}
        if api_style == "responses"
        else {"type": "json_schema", "json_schema": request.response_format}
    )
    assert json.loads(calls[1]["text"]) == expected_format


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_chat_complete_replays_only_chat_reasoning_provenance(field: str) -> None:
    from iris.providers.chat_completions import ChatCompletionsMapper

    mapper = ChatCompletionsMapper()
    response = mapper.parse_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer", field: "old thought"},
                }
            ]
        },
        provider="deepseek",
    )
    message = Msg.model_validate_json(response.to_msg().model_dump_json())
    message.metadata["reasoning"] = "new thought"
    assert mapper.format_messages([message])[0][field] == "new thought"
    assert "old thought" not in json.dumps(message.metadata["chat_completions"])
    other = Msg.assistant(
        "answer", metadata={"reasoning": "Responses reasoning", "responses": {"items": []}}
    )
    assert "reasoning" not in mapper.format_messages([other])[0]
    assert "reasoning_content" not in mapper.format_messages([other])[0]


@pytest.mark.asyncio
async def test_responses_adapter_calls_litellm_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Responses 与 Chat 共用 LiteLLM 作为调用入口。"""
    import litellm

    from iris.providers.responses import ResponsesAdapter

    seen: list[dict[str, Any]] = []
    result = _response()

    async def invoke(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return result

    monkeypatch.setattr(litellm, "aresponses", invoke)
    kwargs = {"model": "openai/gpt-4o", "input": [], "api_key": "test", "num_retries": 0}
    assert await ResponsesAdapter().invoke(kwargs) is result
    assert seen == [kwargs]
