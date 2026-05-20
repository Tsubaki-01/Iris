from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris.observability import (
    LoguruToolLogSink,
    ToolLogEmitter,
    ToolLogEvent,
    ToolLogEventType,
    ToolLogRedactor,
)
from iris.tools import ToolExecutionContext, ToolResult


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[ToolLogEvent] = []

    def emit(self, event: ToolLogEvent) -> None:
        self.events.append(event)


class AsyncCollectingSink:
    def __init__(self) -> None:
        self.events: list[ToolLogEvent] = []

    async def emit(self, event: ToolLogEvent) -> None:
        self.events.append(event)


class BrokenSink:
    def emit(self, event: ToolLogEvent) -> None:
        del event
        raise RuntimeError("sink failed")


def test_tool_log_event_serializes_stable_payload() -> None:
    event = ToolLogEvent(
        event_type=ToolLogEventType.FINISHED,
        timestamp=1.0,
        trace_id="trace_1",
        session_id="session_1",
        agent_id="agent_1",
        call_id="call_1",
        tool_name="read_file",
        tool_group="file",
        capabilities=["read"],
        elapsed_ms=2.5,
        metadata={"param_keys": ["file_path"]},
    )

    payload = event.model_dump(mode="json")

    assert payload == {
        "event_type": "finished",
        "timestamp": 1.0,
        "trace_id": "trace_1",
        "session_id": "session_1",
        "agent_id": "agent_1",
        "call_id": "call_1",
        "tool_name": "read_file",
        "tool_group": "file",
        "capabilities": ["read"],
        "elapsed_ms": 2.5,
        "is_error": False,
        "error_code": "",
        "artifact_path": "",
        "metadata": {"param_keys": ["file_path"]},
    }


def test_tool_log_redactor_summarizes_params_without_values() -> None:
    summary = ToolLogRedactor.params_summary(
        {"query": "hello", "api_key": "secret", "nested": {"token": "hidden"}}
    )

    assert summary == {"param_keys": ["api_key", "nested", "query"]}
    assert "secret" not in str(summary)
    assert "hidden" not in str(summary)


def test_tool_log_redactor_summarizes_result_without_content(tmp_path: Path) -> None:
    result = ToolResult(
        tool_use_id="call_1",
        tool_name="grep_search",
        stats={"elapsed_ms": 1.25},
        metadata={"private": "value"},
    )

    summary = ToolLogRedactor.result_summary(result)

    assert summary == {
        "content_length": 0,
        "has_artifact": False,
        "artifact_size_bytes": 0,
    }
    assert str(tmp_path) not in str(summary)
    assert "private" not in str(summary)


@pytest.mark.asyncio
async def test_tool_log_emitter_sends_to_sync_and_async_sinks(tmp_path: Path) -> None:
    sync_sink = CollectingSink()
    async_sink = AsyncCollectingSink()
    emitter = ToolLogEmitter([sync_sink, async_sink])

    await emitter.emit(
        ToolLogEventType.STARTED,
        context=ToolExecutionContext(workspace_root=tmp_path, call_id="call_1"),
        tool_name="demo",
        metadata={"param_keys": ["value"]},
    )

    assert [event.event_type for event in sync_sink.events] == [ToolLogEventType.STARTED]
    assert [event.event_type for event in async_sink.events] == [ToolLogEventType.STARTED]
    assert sync_sink.events[0].tool_name == "demo"


@pytest.mark.asyncio
async def test_tool_log_emitter_isolates_sink_failures(tmp_path: Path) -> None:
    collecting = CollectingSink()
    emitter = ToolLogEmitter([BrokenSink(), collecting])

    await emitter.emit(
        ToolLogEventType.FINISHED,
        context=ToolExecutionContext(workspace_root=tmp_path, call_id="call_1"),
        tool_name="demo",
    )

    assert [event.event_type for event in collecting.events] == [ToolLogEventType.FINISHED]


def test_loguru_tool_log_sink_accepts_event() -> None:
    sink = LoguruToolLogSink()

    sink.emit(
        ToolLogEvent(
            event_type=ToolLogEventType.STARTED,
            timestamp=1.0,
            call_id="call_1",
            tool_name="demo",
        )
    )
