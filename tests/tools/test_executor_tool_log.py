from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris.message import ToolUseBlock
from iris.observability import ToolLogEmitter, ToolLogEvent, ToolLogEventType
from iris.tools import (
    CircuitBreaker,
    DefaultPermissionPolicy,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    register_file_tools,
)


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[ToolLogEvent] = []

    def emit(self, event: ToolLogEvent) -> None:
        self.events.append(event)


class DenyAllPermissionPolicy:
    def check(self, tool: Any, params: dict[str, Any], context: ToolExecutionContext) -> Any:
        del tool, params, context
        return DefaultPermissionPolicy(allow_writes=False).check(
            _WriteOnlyTool(),
            {},
            ToolExecutionContext(workspace_root=Path(".")),
        )


class _WriteOnlyTool:
    name = "write_file"

    def is_read_only(self, params: dict[str, Any]) -> bool:
        del params
        return False


def _executor_with_log(registry: ToolRegistry) -> tuple[ToolExecutor, CollectingSink]:
    sink = CollectingSink()
    return ToolExecutor(registry, tool_log=ToolLogEmitter([sink])), sink


@pytest.mark.asyncio
async def test_executor_emits_started_and_finished_for_success(tmp_path: Path) -> None:
    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")
    executor, sink = _executor_with_log(registry)

    result = await executor.execute_one(
        ToolUseBlock(id="call_1", name="greet", input={"name": "Iris"}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session_1"),
    )

    assert result.is_error is False
    assert [(event.event_type, event.call_id) for event in sink.events] == [
        (ToolLogEventType.STARTED, "call_1"),
        (ToolLogEventType.FINISHED, "call_1"),
    ]
    assert sink.events[0].metadata == {"param_keys": ["name"]}
    assert sink.events[-1].tool_name == "greet"
    assert sink.events[-1].elapsed_ms is not None


@pytest.mark.asyncio
async def test_executor_emits_failed_for_unknown_tool(tmp_path: Path) -> None:
    executor, sink = _executor_with_log(ToolRegistry())

    result = await executor.execute_one(
        ToolUseBlock(id="call_1", name="missing", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert [(event.event_type, event.error_code) for event in sink.events] == [
        (ToolLogEventType.STARTED, ""),
        (ToolLogEventType.FAILED, "NOT_FOUND"),
    ]


@pytest.mark.asyncio
async def test_executor_emits_failed_for_validation_error(tmp_path: Path) -> None:
    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")
    executor, sink = _executor_with_log(registry)

    result = await executor.execute_one(
        ToolUseBlock(id="call_1", name="greet", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert sink.events[-1].event_type == ToolLogEventType.FAILED
    assert sink.events[-1].error_code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_executor_emits_permission_denied(tmp_path: Path) -> None:
    registry = register_file_tools()
    sink = CollectingSink()
    executor = ToolExecutor(registry, tool_log=ToolLogEmitter([sink]))

    result = await executor.execute_one(
        ToolUseBlock(
            id="call_1",
            name="write_file",
            input={"file_path": "notes.txt", "content": "new"},
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.error is not None
    assert result.error.code == "PERMISSION_ERROR"
    assert sink.events[-1].event_type == ToolLogEventType.PERMISSION_DENIED
    assert sink.events[-1].error_code == "PERMISSION_ERROR"


@pytest.mark.asyncio
async def test_executor_emits_circuit_open(tmp_path: Path) -> None:
    def fail() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register_function(fail, description="失败工具")
    sink = CollectingSink()
    executor = ToolExecutor(
        registry,
        circuit_breaker=CircuitBreaker(failure_threshold=1, cooldown_seconds=60),
        tool_log=ToolLogEmitter([sink]),
    )

    await executor.execute_one(
        ToolUseBlock(id="call_1", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    result = await executor.execute_one(
        ToolUseBlock(id="call_2", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.error is not None
    assert result.error.code == "CIRCUIT_OPEN"
    assert sink.events[-1].event_type == ToolLogEventType.CIRCUIT_OPEN
    assert sink.events[-1].call_id == "call_2"


@pytest.mark.asyncio
async def test_executor_emits_artifact_created(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("\n".join(f"needle {index}" for index in range(80)), encoding="utf-8")
    sink = CollectingSink()
    executor = ToolExecutor(
        register_file_tools(max_result_chars=120),
        artifact_preview_chars=80,
        tool_log=ToolLogEmitter([sink]),
    )

    result = await executor.execute_one(
        ToolUseBlock(id="grep_1", name="grep_search", input={"pattern": "needle"}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session_1"),
    )

    assert result.artifact is not None
    artifact_events = [
        event for event in sink.events if event.event_type == ToolLogEventType.ARTIFACT_CREATED
    ]
    assert len(artifact_events) == 1
    assert artifact_events[0].artifact_path == str(result.artifact.path)


@pytest.mark.asyncio
async def test_executor_tool_log_keeps_concurrent_call_ids_isolated(tmp_path: Path) -> None:
    def echo(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    executor, sink = _executor_with_log(registry)

    await executor.execute_many(
        [
            ToolUseBlock(id="call_1", name="echo", input={"value": "a"}),
            ToolUseBlock(id="call_2", name="echo", input={"value": "b"}),
        ],
        ToolExecutionContext(workspace_root=tmp_path),
    )

    finished_call_ids = {
        event.call_id for event in sink.events if event.event_type == ToolLogEventType.FINISHED
    }
    assert finished_call_ids == {"call_1", "call_2"}
