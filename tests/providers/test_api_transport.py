"""真实 LiteLLM 编码和 HTTP/SSE 传输下的双协议契约。"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.openai import OpenAIChatCompletion
from PIL import Image

from iris.message import (
    LLMRequest,
    ModelResponseCompleted,
    Msg,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
    image_block_from_saved,
)
from iris.providers import create_provider_client
from iris.utils.images import save_image


def _responses_result(model: str) -> dict[str, Any]:
    return {
        "id": "response-wire",
        "created_at": 1,
        "object": "response",
        "model": model,
        "status": "completed",
        "store": False,
        "output": [
            {
                "type": "message",
                "id": "msg-wire",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "checking", "annotations": []}],
            },
            {
                "type": "function_call",
                "id": "fc-wire",
                "call_id": "call-wire",
                "name": "lookup",
                "arguments": '{"q":"Iris"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }


def _chat_result(model: str) -> dict[str, Any]:
    return {
        "id": "response-wire",
        "created": 1,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "call-wire",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q":"Iris"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def _sse(api_style: str, model: str) -> bytes:
    if api_style == "responses":
        final = _responses_result(model)
        call = {**final["output"][1], "arguments": "", "status": "in_progress"}
        events = [
            {
                "type": "response.created",
                "response": {**final, "status": "in_progress", "output": [], "usage": None},
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "item_id": "msg-wire",
                "content_index": 0,
                "delta": "checking",
            },
            {"type": "response.output_item.added", "output_index": 1, "item": call},
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc-wire",
                "delta": '{"q":"Iris"}',
            },
            {"type": "response.completed", "response": final},
        ]
        events = [{**event, "sequence_number": index} for index, event in enumerate(events)]
    else:
        final = _chat_result(model)
        common = {
            "id": final["id"],
            "created": 1,
            "object": "chat.completion.chunk",
            "model": model,
        }
        events = [
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "checking"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, **final["choices"][0]["message"]["tool_calls"][0]}
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {**common, "choices": [], "usage": final["usage"]},
        ]
    data = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if api_style == "chat_completions":
        data += "data: [DONE]\n\n"
    return data.encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("api_style", ["responses", "chat_completions"])
@pytest.mark.parametrize(
    ("provider", "model", "base"),
    [
        ("openai", "gpt-4o", "https://api.openai.com/v1"),
        ("deepseek", "deepseek-flash", "https://api.deepseek.com"),
    ],
)
async def test_protocol_http_and_native_or_simulated_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_images: bool,
    stream: bool,
    api_style: str,
    provider: str,
    model: str,
    base: str,
) -> None:
    import litellm.llms.custom_httpx.llm_http_handler as http_handler

    import iris.providers.factory as factory

    monkeypatch.setattr(factory, "is_config_initialized", lambda: False)

    seen: list[httpx.Request] = []
    picture = BytesIO()
    with Image.new("RGB", (8, 6), color="red") as pixels:
        pixels.save(picture, format="PNG")
    image_bytes = picture.getvalue()
    image = image_block_from_saved(save_image(image_bytes, cache_dir=tmp_path / "images"))

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(api_style, model)
            )
        result = _responses_result(model) if api_style == "responses" else _chat_result(model)
        return httpx.Response(200, json=result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler = object.__new__(AsyncHTTPHandler)
        handler.client = http_client
        handler.timeout = 5
        monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda **kwargs: handler)
        monkeypatch.setattr(
            OpenAIChatCompletion,
            "_get_async_http_client",
            staticmethod(lambda **kwargs: http_client),
        )
        monkeypatch.setattr(
            OpenAIChatCompletion, "get_cached_openai_client", lambda *args, **kwargs: None
        )
        client = create_provider_client(
            f"{provider}/{model}", api_key="wire-test-key", api_style=api_style
        )
        request = LLMRequest(
            model=model,
            stream=stream,
            messages=[
                Msg.user([TextBlock(text="start"), image] if with_images else "start"),
                Msg.assistant("earlier answer"),
                Msg.assistant([ToolUseBlock(id="call-before", name="lookup", input={})]),
                Msg.tool_result(
                    tool_use_id="call-before",
                    name="lookup",
                    content=[TextBlock(text="tool result"), image]
                    if with_images
                    else "tool result",
                ),
            ],
            tools=[
                ToolSpec(
                    name="lookup",
                    description="Search",
                    input_schema={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                )
            ],
            tool_choice={"name": "lookup"},
            response_format={
                "name": "answer",
                "schema": {"type": "object", "properties": {}},
                "strict": True,
            },
            provider_options={"num_retries": 0},
        )
        if stream:
            events = [event async for event in client.stream(request)]
            assert isinstance(events[-1], ModelResponseCompleted), events[-1]
            result = events[-1].response
        else:
            result = await client.complete(request)

    assert result.content == [
        TextBlock(text="checking"),
        ToolUseBlock(id="call-wire", name="lookup", input={"q": "Iris"}),
    ]
    assert result.finish_reason == "tool_calls"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (7, 3, 10)
    assert len(seen) == 1
    endpoint = "/responses" if api_style == "responses" else "/chat/completions"
    assert str(seen[0].url) == base + endpoint
    assert seen[0].headers["authorization"] == "Bearer wire-test-key"
    body = json.loads(seen[0].content)
    assert body["model"] == model
    if api_style == "responses":
        assert "messages" not in body and "previous_response_id" not in body
        assert body["store"] is False
        if stream:
            # 锁定 LiteLLM 为未登记的 OpenAI-compatible 模型模拟流式。
            assert bool(body.get("stream")) is (provider == "openai")
        assert body["input"][1]["content"][0]["type"] == "input_text"
        assert body["input"][-1]["call_id"] == "call-before"
        if with_images:
            user_image = body["input"][0]["content"][1]
            tool_image = body["input"][-1]["output"][1]
            assert user_image == tool_image
            assert user_image["type"] == "input_image" and user_image["detail"] == "high"
            assert base64.b64decode(user_image["image_url"].split(",", 1)[1]) == image_bytes
        assert body["tools"][0]["name"] == "lookup"
        assert body["tool_choice"] == {"type": "function", "name": "lookup"}
        assert body["text"]["format"] == {"type": "json_schema", **request.response_format}
    else:
        assert "input" not in body and "store" not in body
        if with_images:
            assert body["messages"][-2]["role"] == "tool"
            assert body["messages"][-2]["tool_call_id"] == "call-before"
            assert "tool result" in body["messages"][-2]["content"]
            assert body["messages"][-1]["role"] == "user"
            parts = body["messages"][-1]["content"]
            assert "call-before" in json.dumps(parts)
            user_image = body["messages"][0]["content"][1]
            tool_image = next(part for part in parts if part["type"] == "image_url")
            assert user_image == tool_image
            assert user_image["image_url"]["detail"] == "high"
            assert base64.b64decode(user_image["image_url"]["url"].split(",", 1)[1]) == image_bytes
        else:
            assert body["messages"][-1] == {
                "role": "tool",
                "tool_call_id": "call-before",
                "content": "tool result",
            }
        assert body["tools"][0]["function"]["name"] == "lookup"
        assert body["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": request.response_format,
        }
        if stream:
            assert body["stream_options"]["include_usage"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_responses_http_failure_never_switches_protocol(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    import litellm
    import litellm.llms.custom_httpx.llm_http_handler as http_handler

    from iris.exceptions import IrisRateLimitExceededError
    from iris.message import ModelResponseFailed
    from iris.providers import ProviderClient

    seen: list[httpx.Request] = []
    raw_responses: list[httpx.Response] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        result = httpx.Response(
            429,
            json={
                "error": {"message": "rate limit", "type": "rate_limit_error", "code": "rate_limit"}
            },
        )
        raw_responses.append(result)
        return result

    async def no_chat(**kwargs: Any) -> None:
        pytest.fail("Responses failure must never switch to Chat")

    monkeypatch.setattr(litellm, "acompletion", no_chat)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler = object.__new__(AsyncHTTPHandler)
        handler.client = http_client
        handler.timeout = 5
        monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda **kwargs: handler)
        client = ProviderClient(
            provider="openai", api_key="test", base_url="https://native.test/v1"
        )
        request = LLMRequest(
            model="vendor/native-model", stream=stream, provider_options={"num_retries": 0}
        )
        if stream:
            events = [event async for event in client.stream(request)]
            assert len(events) == 1 and isinstance(events[0], ModelResponseFailed)
            assert events[0].error.code == "PROVIDER_ERROR"
            assert events[0].error.retryable is True
        else:
            with pytest.raises(IrisRateLimitExceededError):
                await client.complete(request)
    assert len(seen) == 1
    assert str(seen[0].url) == "https://native.test/v1/responses"
    assert json.loads(seen[0].content)["model"] == "vendor/native-model"
    assert raw_responses[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["consumer_stop", "cancelled", "eof"])
async def test_responses_litellm_stream_closes_http_body(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    import asyncio

    import litellm.llms.custom_httpx.llm_http_handler as http_handler

    from iris.message import ModelResponseFailed
    from iris.providers import ProviderClient

    class Body(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            event = {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    **_responses_result("gpt-4o"),
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
            if ending == "cancelled":
                raise asyncio.CancelledError

        async def aclose(self) -> None:
            self.closed = True

    body = Body()
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler = object.__new__(AsyncHTTPHandler)
        handler.client = http_client
        handler.timeout = 5
        monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda **kwargs: handler)
        client = ProviderClient(
            provider="openai", api_key="test", base_url="https://native.test/v1"
        )
        events = client.stream(
            LLMRequest(model="gpt-4o", stream=True, provider_options={"num_retries": 0})
        )
        assert (await anext(events)).kind == "response.started"
        if ending == "consumer_stop":
            await events.aclose()
        elif ending == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await anext(events)
        else:
            terminal = await anext(events)
            assert isinstance(terminal, ModelResponseFailed)
            assert terminal.error.code == "PROVIDER_STREAM_INTERRUPTED"
            await events.aclose()
        assert body.closed
    assert len(seen) == 1
    payload = json.loads(seen[0].content)
    assert payload["stream"] is True and payload["model"] == "gpt-4o"
