from __future__ import annotations

import pytest

from iris.message import ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    PermissionEffect,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
)


def test_prepare_many_returns_human_gate_without_executing_tool_or_middleware() -> None:
    calls: list[str] = []
    middleware_calls: list[str] = []

    def write_note(content: str) -> str:
        calls.append(content)
        return "written"

    class Middleware:
        def before_call(self, *args: object) -> None:
            del args
            middleware_calls.append("before")

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        middleware=[Middleware()],
    )
    context = ToolExecutionContext(workspace_root=".", session_id="session_1")

    plan = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={"content": "hello"})],
        context,
    )

    prepared = plan.calls[0]
    assert prepared.permission is not None
    assert prepared.permission.effect is PermissionEffect.REQUIRE_HUMAN
    assert prepared.human_request is not None
    assert calls == []
    assert middleware_calls == []


@pytest.mark.asyncio
async def test_execute_prepared_allows_only_matching_human_approval() -> None:
    calls: list[str] = []

    def write_note(content: str) -> str:
        calls.append(content)
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
    )
    context = ToolExecutionContext(workspace_root=".")
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={"content": "hello"})],
        context,
    ).calls[0]

    rejected = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="other_call",
    )
    approved = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="call_1",
    )

    assert rejected.is_error is True
    assert approved.is_error is False
    assert calls == ["hello"]


@pytest.mark.asyncio
async def test_execute_prepared_never_allows_denied_call() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("called")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode="deny"),
    )
    context = ToolExecutionContext(workspace_root=".")
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={})], context
    ).calls[0]

    result = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="call_1",
    )

    assert result.is_error is True
    assert calls == []
