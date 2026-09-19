"""记忆写工具沿既有 WRITE、HITL 和 claim 路径执行。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunEventKind, RunStopReason, ToolCallPhase
from iris.memory import (
    MemoryAccessPolicy,
    MemoryEvent,
    MemoryIOExecutionMode,
    MemoryItem,
    MemoryService,
    SQLiteMemoryStore,
    register_memory_tools,
)
from iris.message import ToolUseBlock
from iris.store import SQLiteStore
from iris.tools.permissions import DefaultPermissionPolicy

from .fakes import StaticProvider, build_runtime, text_response, tool_response


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_memory_write_waits_for_confirmation_and_resumes_once(
    tmp_path: Path, approved: bool
) -> None:
    """确认前及拒绝后均不写库，批准后通过 durable claim 只写一次。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    registry = register_memory_tools(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(),
        tool_names=("memory.remember",),
    )
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="remember",
                name="memory_remember",
                input={"text": "用户偏好中文", "reason": "用户要求记住"},
            )
        ),
        text_response(),
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider), store=store
    )
    waiting = await runner.start(AgentRunRequest(input="保存偏好", run_id="remember-run"))

    assert waiting.pending_interaction is not None, waiting.error
    assert waiting.pending_interaction.tool_call_id == "remember"
    assert service.list_items(["project"]) == []
    assert RunEventKind.TOOL_CALL_CLAIMED not in {
        event.kind for event in store.list_events("remember-run")
    }
    completed = await runner.resume(
        "remember-run",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve" if approved else "reject"),
    )

    assert completed.run.stop_reason is RunStopReason.COMPLETED, completed.error
    assert len(service.list_items(["project"])) == int(approved)
    [call] = store.list_tool_calls("remember-run")
    assert call.phase is ToolCallPhase.COMMITTED
    assert call.result is not None and call.result.is_error is (not approved)
    assert [event.kind for event in store.list_events("remember-run")].count(
        RunEventKind.TOOL_CALL_CLAIMED
    ) == int(approved)
    assert await runner.recover("remember-run") == completed
    assert len(service.list_items(["project"])) == int(approved)


@pytest.mark.asyncio
async def test_cancelled_thread_write_keeps_unknown_outcome_and_does_not_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后台数据库写入不会随协程取消回滚，已 claim 的不确定结果必须保持 unknown。"""
    memory_store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(memory_store, io_execution_mode=MemoryIOExecutionMode.THREAD)
    registry = register_memory_tools(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(),
        tool_names=("memory.remember",),
    )
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="remember",
                name="memory_remember",
                input={"text": "已在后台保存", "reason": "用户要求"},
            )
        )
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runtime = build_runtime(
        tmp_path,
        registry=registry,
        provider=provider,
        permission_policy=DefaultPermissionPolicy(write_mode="allow"),
    )
    runner = AgentRunner(runtime=runtime, store=store)
    committed = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = memory_store.add_item

    def delayed_return(item: MemoryItem, *, event: MemoryEvent) -> MemoryItem:
        stored = original(item, event=event)
        committed.set()
        release.wait(timeout=5)
        finished.set()
        return stored

    monkeypatch.setattr(memory_store, "add_item", delayed_return)
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input="记住事实", run_id="thread-write"))
    )
    try:
        assert await asyncio.to_thread(committed.wait, 2)
        cancelled = await runner.cancel("thread-write", settlement_timeout=2)
        assert await running == cancelled
        assert cancelled.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
        [call] = store.list_tool_calls("thread-write")
        assert call.phase is ToolCallPhase.OUTCOME_UNKNOWN
        events = store.list_events("thread-write")
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)

    assert store.list_events("thread-write") == events
    assert len(service.list_items(["project"])) == 1
    assert await runner.recover("thread-write") == cancelled
    assert len(service.list_items(["project"])) == 1
