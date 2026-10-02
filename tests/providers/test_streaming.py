"""原生 Responses 流式事件与终态解析契约。"""

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
    ModelUsageUpdated,
    Msg,
    TextBlock,
    ToolUseBlock,
)
from iris.providers import ProviderClient


class _RawStream(AsyncIterator[dict[str, Any]]):
    """逐项返回 Responses 事件并记录关闭次数。"""

    def __init__(self, *items: dict[str, Any] | BaseException) -> None:
        self._items = list(items)
        self.close_calls = 0
        self.reads = 0

    def __aiter__(self) -> _RawStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._items:
            raise StopAsyncIteration
        self.reads += 1
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.close_calls += 1


def _response(*output: dict[str, Any], status: str = "completed") -> dict[str, Any]:
    return {
        "id": "resp-1",
        "object": "response",
        "model": "gpt-4o",
        "status": status,
        "output": list(output),
        "usage": {
            "input_tokens": 4,
            "output_tokens": 5,
            "total_tokens": 9,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg-1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call(item_id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _event(kind: str, **values: Any) -> dict[str, Any]:
    return {"type": f"response.{kind}", **values}


def _install_stream(monkeypatch: pytest.MonkeyPatch, raw: _RawStream) -> dict[str, Any]:
    import iris.providers.client as provider_client

    seen: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> _RawStream:
        seen.update(kwargs)
        return raw

    async def no_chat(**kwargs: Any) -> None:
        pytest.fail("流式主链不得调用 Chat Completion")

    monkeypatch.setattr(
        provider_client.ResponsesAdapter,
        "invoke",
        staticmethod(lambda kwargs: fake_aresponses(**kwargs)),
    )
    monkeypatch.setattr(provider_client.litellm, "acompletion", no_chat)
    return seen


async def _collect() -> list[Any]:
    client = ProviderClient(provider="openai", api_key="test-key")
    request = LLMRequest(model="gpt-4o", messages=[Msg.user("你好")], stream=True)
    return [event async for event in client.stream(request)]


@pytest.mark.asyncio
async def test_provider_stream_text_typed_events_and_terminal_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(
        _event("created", response=_response(status="in_progress")),
        _event("in_progress", response=_response(status="in_progress")),
        _event("output_item.added", output_index=0, item=_message("")),
        _event(
            "content_part.added",
            output_index=0,
            item_id="msg-1",
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        ),
        _event("output_text.delta", output_index=0, item_id="msg-1", content_index=0, delta="你"),
        _event("output_text.delta", output_index=0, item_id="msg-1", content_index=0, delta="好"),
        _event("output_text.done", output_index=0, item_id="msg-1", content_index=0, text="你好"),
        _event("output_item.done", output_index=0, item=_message("你好")),
        _event("completed", response=_response(_message("你好"))),
        RuntimeError("terminal 后不得继续拉取"),
    )
    seen = _install_stream(monkeypatch, raw)
    events = await _collect()

    assert seen["stream"] is True
    assert seen["store"] is False
    assert "stream_options" not in seen
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.kind for event in events] == [
        "response.started",
        "block.started",
        "block.delta",
        "block.delta",
        "block.completed",
        "usage.updated",
        "response.completed",
    ]
    terminal = events[-1]
    assert isinstance(terminal, ModelResponseCompleted)
    assert terminal.response.to_msg().text == "你好"
    assert terminal.response.finish_reason == "stop"
    assert (
        terminal.response.input_tokens,
        terminal.response.output_tokens,
        terminal.response.total_tokens,
    ) == (4, 5, 9)
    assert raw.close_calls == 1
    assert raw.reads == 9


@pytest.mark.asyncio
async def test_provider_stream_final_output_supplies_full_reasoning_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = {
        "type": "reasoning",
        "id": "reason-1",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "完整推理"}],
        "encrypted_content": "opaque-replay-data",
    }
    final = _response(reasoning, _message("完整回答"), _call("fc-1", "call-1", "lookup", "{}"))
    raw = _RawStream(
        _event(
            "reasoning_text.delta",
            output_index=0,
            item_id="reason-1",
            content_index=0,
            delta="完整",
        ),
        _event(
            "reasoning_text.done",
            output_index=0,
            item_id="reason-1",
            content_index=0,
            text="完整推理",
        ),
        _event("completed", response=final),
    )
    _install_stream(monkeypatch, raw)
    events = await _collect()

    from iris.providers.responses import ResponsesMapper

    assert events[-1].response == ResponsesMapper().parse_response(final, provider="openai")
    assert events[-1].response.content == [
        TextBlock(text="完整回答"),
        ToolUseBlock(id="call-1", name="lookup", input={}),
    ]
    assert "opaque-replay-data" in events[-1].response.model_dump_json()
    thinking = [event for event in events if isinstance(event, ModelBlockDelta)]
    assert [event.channel for event in thinking] == ["thinking", "thinking"]
    assert thinking[-1].snapshot == "完整推理"


@pytest.mark.asyncio
async def test_provider_stream_parallel_tool_arguments_use_call_id_and_final_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(
        _event("output_item.added", output_index=0, item=_call("fc-a", "call-a", "lookup", "")),
        _event("output_item.added", output_index=1, item=_call("fc-b", "call-b", "list", "")),
        _event("function_call_arguments.delta", output_index=0, item_id="fc-a", delta='{"query":'),
        _event("function_call_arguments.delta", output_index=1, item_id="fc-b", delta="{}"),
        _event("function_call_arguments.delta", output_index=0, item_id="fc-a", delta='"Iris"}'),
        _event(
            "function_call_arguments.done",
            output_index=0,
            item_id="fc-a",
            name="lookup",
            arguments='{"query":"Iris"}',
        ),
        _event(
            "output_item.done",
            output_index=0,
            item=_call("fc-a", "call-a", "lookup", '{"query":"Iris"}'),
        ),
        _event(
            "completed",
            response=_response(
                _call("fc-a", "call-a", "lookup", '{"query":"Iris"}'),
                _call("fc-b", "call-b", "list", "{}"),
            ),
        ),
    )
    _install_stream(monkeypatch, raw)
    events = await _collect()

    arguments = [
        event
        for event in events
        if isinstance(event, ModelBlockDelta) and event.channel == "tool_arguments"
    ]
    assert [(event.block.tool_call_id, event.snapshot) for event in arguments] == [
        ("call-a", '{"query":'),
        ("call-b", "{}"),
        ("call-a", '{"query":"Iris"}'),
    ]
    assert arguments[0].block.block_id != arguments[0].block.tool_call_id
    assert [call.id for call in events[-1].response.to_msg().tool_calls] == ["call-a", "call-b"]
    assert events[-1].response.finish_reason == "tool_calls"
    assert len([event for event in events if event.kind == "block.completed"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_provider_stream_failure_reports_usage_without_committing_tools(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    final = _response(_call("fc-1", "call-1", "lookup", "{"), status=status)
    final["incomplete_details"] = {"reason": "max_output_tokens"}
    final["error"] = {"code": "server_error", "message": "处理失败"}
    raw = _RawStream(_event(status, response=final))
    _install_stream(monkeypatch, raw)
    events = await _collect()

    assert isinstance(events[-1], ModelResponseFailed)
    assert not any(isinstance(event, ModelResponseCompleted) for event in events)
    usage = [event for event in events if isinstance(event, ModelUsageUpdated)]
    assert len(usage) == 1
    assert usage[0].usage.total_tokens == 9
    assert usage[0].usage.complete is True
    assert status in events[-1].error.message
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_provider_stream_error_event_is_terminal_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream({"type": "error", "code": "server_error", "message": "处理失败"})
    _install_stream(monkeypatch, raw)
    events = await _collect()
    assert isinstance(events[-1], ModelResponseFailed)
    assert "server_error" in events[-1].error.message
    assert "处理失败" in events[-1].error.message
    assert not any(event.kind == "usage.updated" for event in events)
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_provider_stream_eof_after_output_item_done_is_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(
        _event("output_item.added", output_index=0, item=_call("fc-1", "call-1", "lookup", "")),
        _event("function_call_arguments.delta", output_index=0, item_id="fc-1", delta="{}"),
        _event("output_item.done", output_index=0, item=_call("fc-1", "call-1", "lookup", "{}")),
    )
    _install_stream(monkeypatch, raw)
    events = await _collect()
    assert isinstance(events[-1], ModelResponseFailed)
    assert events[-1].error.code == "PROVIDER_STREAM_INTERRUPTED"
    assert events[-1].semantic_output_emitted is True
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_provider_stream_rejects_malformed_final_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(
        _event("completed", response=_response(_call("fc-1", "call-1", "lookup", "{")))
    )
    _install_stream(monkeypatch, raw)
    events = await _collect()
    assert isinstance(events[-1], ModelResponseFailed)
    assert not any(isinstance(event, ModelResponseCompleted) for event in events)
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_provider_stream_reasoning_summary_keeps_separate_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(
        _event(
            "reasoning_summary_text.delta",
            output_index=0,
            item_id="reason-1",
            summary_index=0,
            delta="先核对",
        ),
        _event(
            "reasoning_summary_text.done",
            output_index=0,
            item_id="reason-1",
            summary_index=0,
            text="先核对",
        ),
        _event(
            "reasoning_summary_text.delta",
            output_index=0,
            item_id="reason-1",
            summary_index=1,
            delta="再回答",
        ),
        _event(
            "completed",
            response=_response(
                {
                    "type": "reasoning",
                    "id": "reason-1",
                    "encrypted_content": "opaque",
                    "summary": [
                        {"type": "summary_text", "text": "先核对"},
                        {"type": "summary_text", "text": "再回答"},
                    ],
                },
                _message("完成"),
            ),
        ),
    )
    _install_stream(monkeypatch, raw)
    events = await _collect()
    deltas = [event for event in events if isinstance(event, ModelBlockDelta)]
    assert [event.snapshot for event in deltas] == ["先核对", "再回答"]
    assert deltas[0].block.block_id != deltas[1].block.block_id
    assert events[-1].response.reasoning == "先核对\n再回答"


@pytest.mark.asyncio
async def test_provider_stream_maps_raw_failure_after_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(
        _event("output_text.delta", output_index=0, item_id="msg-1", content_index=0, delta="部分"),
        RuntimeError("raw-secret"),
    )
    _install_stream(monkeypatch, raw)
    events = await _collect()
    assert isinstance(events[-1], ModelResponseFailed)
    assert events[-1].semantic_output_emitted is True
    assert "raw-secret" not in events[-1].error.message
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_provider_stream_propagates_local_cancellation_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(asyncio.CancelledError())
    _install_stream(monkeypatch, raw)
    with pytest.raises(asyncio.CancelledError):
        await _collect()
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_provider_stream_closes_when_consumer_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawStream(_event("created", response=_response(status="in_progress")))
    _install_stream(monkeypatch, raw)
    stream = ProviderClient(provider="openai", api_key="test-key").stream(
        LLMRequest(model="gpt-4o", stream=True)
    )
    assert (await anext(stream)).kind == "response.started"
    await stream.aclose()
    assert raw.close_calls == 1
