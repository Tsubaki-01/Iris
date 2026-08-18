"""Waiting interaction expiry 与 run deadline 的惰性结算测试。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunStopReason
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import FrozenClock, StaticProvider, build_runtime, tool_response


@pytest.mark.asyncio
async def test_resume_at_interaction_expiry_terminalizes_without_tool_effect(
    tmp_path: Path,
) -> None:
    effects: list[str] = []

    def write(value: str) -> str:
        effects.append(value)
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    clock = FrozenClock()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write", name="write", input={"value": "x"}))
            ),
        ),
        store=store,
        clock=clock,
    )
    waiting = await runner.start(
        AgentRunRequest(input="写入", run_id="run-expiry"),
        options=AgentRunOptions(limits=RunLimits(interaction_timeout_seconds=10)),
    )
    assert waiting.pending_interaction is not None
    clock.advance(seconds=10)
    assert runner.get_result("run-expiry").run.stop_reason is None

    result = await runner.resume(
        "run-expiry",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )

    assert result.run.stop_reason is RunStopReason.INTERACTION_EXPIRED
    assert effects == []
    [tool_result] = store.load_session("default").messages[-1].tool_results
    assert tool_result.tool_use_id == "write"
    assert tool_result.metadata["error"]["code"] == "TOOL_NOT_STARTED"


@pytest.mark.asyncio
async def test_earlier_run_deadline_wins_over_later_interaction_expiry(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def write(value: str) -> str:
        return value

    registry.register_function(
        write,
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    clock = FrozenClock()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write", name="write", input={"value": "x"}))
            ),
        ),
        store=store,
        clock=clock,
    )
    waiting = await runner.start(
        AgentRunRequest(input="写入", run_id="run-deadline"),
        options=AgentRunOptions(
            limits=RunLimits(
                deadline_at=clock.now() + timedelta(seconds=5),
                interaction_timeout_seconds=10,
            )
        ),
    )
    assert waiting.pending_interaction is not None
    clock.advance(seconds=10)

    result = await runner.resume(
        "run-deadline",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="reject"),
    )

    assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    [tool_result] = store.load_session("default").messages[-1].tool_results
    assert tool_result.tool_use_id == "write"
    assert tool_result.metadata["error"]["code"] == "TOOL_NOT_STARTED"
