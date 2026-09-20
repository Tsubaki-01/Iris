"""AgentRunner expected-fence recovery 决策矩阵测试。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.harness import AgentRunner
from iris.harness._commit_port import StoreRuntimeCommitPort
from iris.harness._events import _RunEventCollector
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    ClaimToolCall,
    FinishRun,
    RunCommit,
    RunEventKind,
    RunLimits,
    RunStopReason,
    SessionCompaction,
    SessionContextWindow,
    TokenUsage,
    ToolCallPhase,
)
from iris.message import ToolUseBlock
from iris.providers.openai import OpenAIChatMapper
from iris.runtime import RuntimeCompactionCommit, RuntimeCursor
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import (
    BlockingProvider,
    CountingAgentRuntime,
    FrozenClock,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


@pytest.mark.asyncio
async def test_safe_recovery_reuses_reserved_model_step_and_executes_once(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    await AgentRunner(
        runtime=build_runtime(tmp_path, provider=StaticProvider(text_response("前一轮"))),
        store=store,
    ).start(AgentRunRequest(input="已有历史"))
    first = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    running = asyncio.create_task(
        first.start(AgentRunRequest(input="恢复", run_id="run-safe-recover"))
    )
    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    crashed = store.load_run("run-safe-recover")
    assert crashed is not None and crashed.current_activation_id is not None
    checkpoint = store.load_checkpoint(crashed.run_id)
    assert checkpoint is not None
    port = StoreRuntimeCommitPort(
        store=store,
        run=crashed,
        activation_id=crashed.current_activation_id,
        cursor=RuntimeCursor.model_validate(checkpoint.engine_cursor),
        clock=first._now,
        event_collector=_RunEventCollector(),
        workspace_root=tmp_path,
    )
    session = port.load_session()
    summary = SessionCompaction(summary="前一轮已完成", covered_message_count=2)
    port.record_compaction_usage(TokenUsage(total_tokens=22_000))
    port.commit_compaction(
        RuntimeCompactionCommit(
            cursor_before=port.cursor,
            expected_session_revision=session.revision,
            compaction=summary,
            context_window=SessionContextWindow(),
            before_input_tokens=80_000,
            after_input_tokens=20_000,
        )
    )

    runtime = CountingAgentRuntime(
        build_runtime(tmp_path, provider=StaticProvider(text_response("已恢复")))
    )
    second = AgentRunner(runtime=runtime, store=store)
    result = await second.recover(
        "run-safe-recover",
        expected_activation_id=crashed.current_activation_id,
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert runtime.execute_calls == 1
    assert runtime.activations[0].run_input == "恢复"
    assert runtime.activations[0].initial_session_message_count == 2
    assert result.run.usage.compaction.total_tokens == 22_000
    recovered_session = store.load_session("default")
    assert recovered_session.compaction == summary
    assert sum(message.text == "恢复" for message in recovered_session.messages) == 1
    events = store.list_events("run-safe-recover")
    assert [event.kind for event in events].count(RunEventKind.MODEL_STEP_RESERVED) == 1
    assert [event.kind for event in events].count(RunEventKind.ACTIVATION_ABANDONED) == 1


@pytest.mark.asyncio
async def test_sqlite_restart_discovers_active_lane_and_recovers_with_exact_fence(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    path = tmp_path / "restart-lifecycle.db"
    first_store = SQLiteStore(path)
    running = asyncio.create_task(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=first_store,
        ).start(AgentRunRequest(input="重启恢复", run_id="run-restart-discovery"))
    )
    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    restarted_store = SQLiteStore(path)
    discovered_run_id = restarted_store.load_session_lane("default")
    assert discovered_run_id == "run-restart-discovery"
    discovered = restarted_store.load_run(discovered_run_id)
    assert discovered is not None and discovered.current_activation_id is not None

    recovery_provider = StaticProvider(text_response("恢复完成"))
    result = await AgentRunner(
        runtime=build_runtime(tmp_path, provider=recovery_provider),
        store=restarted_store,
    ).recover(
        discovered_run_id,
        expected_activation_id=discovered.current_activation_id,
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert len(recovery_provider.requests) == 1
    assert restarted_store.load_session_lane("default") is None


@pytest.mark.asyncio
async def test_outcome_ready_recovery_finalizes_without_provider_call(tmp_path: Path) -> None:
    class FailFinishOnceStore(InMemoryLifecycleStore):
        failed = False

        def finish_run(self, command: FinishRun) -> RunCommit:
            if not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("finish crash")
            return super().finish_run(command)

    store = FailFinishOnceStore()
    first = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    with pytest.raises(IrisRunPersistenceError, match="finish crash"):
        await first.start(AgentRunRequest(input="完成", run_id="run-outcome-ready"))
    crashed = store.load_run("run-outcome-ready")
    assert crashed is not None and crashed.current_activation_id is not None
    provider = StaticProvider()

    result = await AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
    ).recover(
        "run-outcome-ready",
        expected_activation_id=crashed.current_activation_id,
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.assistant_message is not None
    assert provider.requests == []


@pytest.mark.asyncio
async def test_recovery_marks_unresolved_claim_unknown_without_replaying_tool(
    tmp_path: Path,
) -> None:
    class CrashAfterClaimStore(InMemoryLifecycleStore):
        def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
            super().claim_tool_call(command)
            raise IrisRunPersistenceError("claim committed before crash")

    effects: list[str] = []
    registry = ToolRegistry()
    registry.register_function(
        lambda: effects.append("effect") or "effect",
        name="effect",
        description="副作用",
    )
    store = CrashAfterClaimStore()
    first = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="effect-1", name="effect", input={}))
            ),
        ),
        store=store,
    )
    with pytest.raises(IrisRunPersistenceError, match="claim committed"):
        await first.start(AgentRunRequest(input="执行", run_id="run-claim-recover"))
    crashed = store.load_run("run-claim-recover")
    assert crashed is not None and crashed.current_activation_id is not None

    result = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry),
        store=store,
    ).recover(
        "run-claim-recover",
        expected_activation_id=crashed.current_activation_id,
    )

    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert result.error is not None and result.error.code == "TOOL_OUTCOME_UNKNOWN"
    assert effects == []
    [record] = store.list_tool_calls("run-claim-recover")
    assert record.phase is ToolCallPhase.OUTCOME_UNKNOWN
    assert [event.kind for event in store.list_events("run-claim-recover")][-3:] == [
        RunEventKind.ACTIVATION_ABANDONED,
        RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
        RunEventKind.RUN_TERMINAL,
    ]

    follow_up_provider = StaticProvider(text_response("继续完成"))
    follow_up = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=follow_up_provider),
        store=store,
    ).start(AgentRunRequest(input="继续", run_id="run-after-claim-recover"))

    assert follow_up.run.stop_reason is RunStopReason.COMPLETED
    wire_messages = OpenAIChatMapper().format_messages(follow_up_provider.requests[0].messages)
    tool_call_ids = [
        call["id"] for message in wire_messages for call in message.get("tool_calls", [])
    ]
    tool_result_ids = [
        message["tool_call_id"] for message in wire_messages if message["role"] == "tool"
    ]
    assert tool_call_ids == ["effect-1"]
    assert tool_result_ids == tool_call_ids
    tool_call_index = next(
        index for index, message in enumerate(wire_messages) if message.get("tool_calls")
    )
    tool_result_index = next(
        index for index, message in enumerate(wire_messages) if message["role"] == "tool"
    )
    next_user_index = next(
        index
        for index, message in enumerate(wire_messages)
        if message["role"] == "user" and message["content"] == "继续"
    )
    assert tool_call_index < tool_result_index < next_user_index


@pytest.mark.asyncio
async def test_recover_terminal_is_idempotent_and_waiting_requires_resume(
    tmp_path: Path,
) -> None:
    terminal_runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=InMemoryLifecycleStore(),
    )
    terminal = await terminal_runner.start(
        AgentRunRequest(input="完成", run_id="run-terminal-recover")
    )
    assert await terminal_runner.recover("run-terminal-recover") == terminal

    registry = ToolRegistry()
    registry.register_function(
        lambda: "write",
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    waiting_runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write-1", name="write", input={}))
            ),
        ),
        store=InMemoryLifecycleStore(),
    )
    await waiting_runner.start(AgentRunRequest(input="等待", run_id="run-wait-recover"))
    with pytest.raises(IrisRunStateError, match="resume"):
        await waiting_runner.recover("run-wait-recover")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deadline_seconds", "interaction_timeout_seconds", "expected_reason"),
    [
        (5, 10, RunStopReason.DEADLINE_EXCEEDED),
        (None, 5, RunStopReason.INTERACTION_EXPIRED),
    ],
)
async def test_waiting_recovery_deterministically_settles_due_time(
    tmp_path: Path,
    deadline_seconds: int | None,
    interaction_timeout_seconds: int,
    expected_reason: RunStopReason,
) -> None:
    clock = FrozenClock()
    registry = ToolRegistry()
    registry.register_function(
        lambda: "write",
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write-due", name="write", input={}))
            ),
        ),
        store=InMemoryLifecycleStore(),
        clock=clock,
    )
    deadline_at = (
        clock.now() + timedelta(seconds=deadline_seconds) if deadline_seconds is not None else None
    )
    waiting = await runner.start(
        AgentRunRequest(
            input="等待",
            run_id=f"run-wait-{expected_reason.value}",
        ),
        options=AgentRunOptions(
            limits=RunLimits(
                deadline_at=deadline_at,
                interaction_timeout_seconds=interaction_timeout_seconds,
            )
        ),
    )
    assert waiting.pending_interaction is not None

    clock.advance(seconds=6)
    result = await runner.recover(waiting.run.run_id)

    assert result.run.stop_reason is expected_reason
    interaction = runner.store.load_interaction(waiting.pending_interaction.interaction_id)
    assert interaction is not None and interaction.status.value == "closed"


@pytest.mark.asyncio
async def test_concurrent_recovery_allows_only_one_activation_takeover(
    tmp_path: Path,
) -> None:
    crashed_provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    first = AgentRunner(
        runtime=build_runtime(tmp_path, provider=crashed_provider),
        store=store,
    )
    running = asyncio.create_task(
        first.start(AgentRunRequest(input="恢复", run_id="run-concurrent-recover"))
    )
    await crashed_provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    crashed = store.load_run("run-concurrent-recover")
    assert crashed is not None and crashed.current_activation_id is not None

    recovering_provider = BlockingProvider(text_response("接管完成"))
    winner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=recovering_provider),
        store=store,
    )
    loser = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    recovery = asyncio.create_task(
        winner.recover(
            "run-concurrent-recover",
            expected_activation_id=crashed.current_activation_id,
        )
    )
    await recovering_provider.started.wait()

    with pytest.raises(IrisRunConflictError, match="fence"):
        await loser.recover(
            "run-concurrent-recover",
            expected_activation_id=crashed.current_activation_id,
        )

    recovering_provider.release.set()
    result = await recovery
    assert result.run.stop_reason is RunStopReason.COMPLETED


@pytest.mark.asyncio
async def test_recovery_uses_current_system_configuration(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    first = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, system_text="原始配置"),
        store=store,
    )
    running = asyncio.create_task(
        first.start(AgentRunRequest(input="恢复", run_id="run-config-change"))
    )
    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    before_run = store.load_run("run-config-change")
    assert before_run is not None and before_run.current_activation_id is not None

    current_provider = StaticProvider(text_response())
    current = AgentRunner(
        runtime=build_runtime(tmp_path, system_text="已变更配置", provider=current_provider),
        store=store,
    )
    result = await current.recover(
        "run-config-change",
        expected_activation_id=before_run.current_activation_id,
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert len(current_provider.requests) == 1
    assert "已变更配置" in current_provider.requests[0].messages[0].text
    assert "原始配置" not in current_provider.requests[0].messages[0].text
