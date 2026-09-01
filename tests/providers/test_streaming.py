"""LiteLLM Chat streaming provider 边界测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from iris.message import (
    LLMRequest,
    ModelBlockDelta,
    ModelResponseCompleted,
    ModelResponseFailed,
    Msg,
)
from iris.providers import ProviderClient


class _RawStream(AsyncIterator[dict[str, Any]]):
    """按脚本返回 LiteLLM-shaped chunks 的测试流。"""

    def __init__(self, *items: dict[str, Any] | BaseException) -> None:
        self._items = list(items)
        self.close_calls = 0

    def __aiter__(self) -> _RawStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._items:
            raise StopAsyncIteration
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.close_calls += 1


def _chunk(
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    choices: bool = True,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stream-1",
        "model": "gpt-4o",
        "object": "chat.completion.chunk",
        "choices": (
            [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}] if choices else []
        ),
        "usage": usage,
    }


async def _collect(client: ProviderClient, request: LLMRequest) -> list[Any]:
    return [event async for event in client.stream(request)]


@pytest.mark.asyncio
async def test_provider_client_stream_aggregates_text_and_usage_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_stream = _RawStream(
        _chunk(delta={"role": "assistant", "content": "你"}),
        _chunk(delta={"content": "好"}),
        _chunk(finish_reason="stop"),
        _chunk(
            choices=False,
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        ),
    )
    seen_kwargs: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _RawStream:
        seen_kwargs.update(kwargs)
        return raw_stream

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)
    client = ProviderClient(provider="openai", api_key="test-key")

    events = await _collect(
        client,
        LLMRequest(model="gpt-4o", messages=[Msg.user("你好")], stream=True),
    )

    assert seen_kwargs["stream"] is True
    assert seen_kwargs["stream_options"] == {"include_usage": True}
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.kind for event in events] == [
        "response.started",
        "block.started",
        "block.delta",
        "block.delta",
        "block.completed",
        "usage.updated",
        "usage.updated",
        "response.completed",
    ]
    terminal = events[-1]
    assert isinstance(terminal, ModelResponseCompleted)
    assert terminal.response.to_msg().text == "你好"
    assert terminal.response.finish_reason == "stop"
    assert terminal.response.input_tokens == 2
    assert terminal.response.output_tokens == 1
    assert terminal.response.total_tokens == 3
    assert raw_stream.close_calls == 1


@pytest.mark.asyncio
async def test_provider_client_stream_aggregates_split_tool_call_only_at_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_stream = _RawStream(
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "look", "arguments": '{"query":'},
                    }
                ]
            }
        ),
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"name": "up", "arguments": '"Iris"}'},
                    }
                ]
            }
        ),
        _chunk(finish_reason="tool_calls"),
    )

    async def fake_acompletion(**kwargs: Any) -> _RawStream:
        return raw_stream

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)

    events = await _collect(
        ProviderClient(provider="openai", api_key="test-key"),
        LLMRequest(model="gpt-4o", stream=True),
    )

    argument_deltas = [
        event
        for event in events
        if isinstance(event, ModelBlockDelta) and event.channel == "tool_arguments"
    ]
    assert [event.snapshot for event in argument_deltas] == [
        '{"query":',
        '{"query":"Iris"}',
    ]
    terminal = events[-1]
    assert isinstance(terminal, ModelResponseCompleted)
    [tool_call] = terminal.response.to_msg().tool_calls
    assert tool_call.id == "call-1"
    assert tool_call.name == "lookup"
    assert tool_call.input == {"query": "Iris"}


@pytest.mark.asyncio
async def test_provider_client_stream_maps_eof_without_finish_to_failed_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_stream = _RawStream(_chunk(delta={"content": "未完成"}))

    async def fake_acompletion(**kwargs: Any) -> _RawStream:
        return raw_stream

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)

    events = await _collect(
        ProviderClient(provider="openai", api_key="test-key"),
        LLMRequest(model="gpt-4o", stream=True),
    )

    terminal = events[-1]
    assert isinstance(terminal, ModelResponseFailed)
    assert terminal.error.code == "PROVIDER_STREAM_INTERRUPTED"
    assert terminal.semantic_output_emitted is True
    assert raw_stream.close_calls == 1


@pytest.mark.asyncio
async def test_provider_client_stream_rejects_malformed_completed_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_stream = _RawStream(
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-invalid",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{"},
                    }
                ]
            }
        ),
        _chunk(finish_reason="tool_calls"),
    )

    async def fake_acompletion(**kwargs: Any) -> _RawStream:
        return raw_stream

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)

    events = await _collect(
        ProviderClient(provider="openai", api_key="test-key"),
        LLMRequest(model="gpt-4o", stream=True),
    )

    terminal = events[-1]
    assert isinstance(terminal, ModelResponseFailed)
    assert terminal.error.code == "PROVIDER_STREAM_PROTOCOL_ERROR"
    assert all(event.kind != "block.completed" for event in events)


@pytest.mark.asyncio
async def test_provider_client_stream_maps_raw_failure_after_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_stream = _RawStream(
        _chunk(delta={"content": "部分"}),
        RuntimeError("raw-secret"),
    )

    async def fake_acompletion(**kwargs: Any) -> _RawStream:
        return raw_stream

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)

    events = await _collect(
        ProviderClient(provider="openai", api_key="test-key"),
        LLMRequest(model="gpt-4o", stream=True),
    )

    terminal = events[-1]
    assert isinstance(terminal, ModelResponseFailed)
    assert terminal.semantic_output_emitted is True
    assert "raw-secret" not in terminal.error.message
    assert raw_stream.close_calls == 1


@pytest.mark.asyncio
async def test_provider_client_stream_propagates_local_cancellation_and_closes_raw_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.providers.client as provider_client

    raw_stream = _RawStream(asyncio.CancelledError())

    async def fake_acompletion(**kwargs: Any) -> _RawStream:
        return raw_stream

    monkeypatch.setattr(provider_client.litellm, "acompletion", fake_acompletion)

    with pytest.raises(asyncio.CancelledError):
        await _collect(
            ProviderClient(provider="openai", api_key="test-key"),
            LLMRequest(model="gpt-4o", stream=True),
        )

    assert raw_stream.close_calls == 1
