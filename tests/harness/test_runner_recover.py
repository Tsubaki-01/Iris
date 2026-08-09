"""AgentRunner expected-fence recovery 决策矩阵测试。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    ClaimToolCall,
    FinishRun,
    RunCommit,
    RunEventKind,
    RunLimits,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import (
    BlockingProvider,
    CountingAgentRuntime,
    FrozenClock,
    StaticProvider,
    build_runtime,
    text_response,
    tool_batch_response,
    tool_response,
)


@pytest.mark.asyncio
async def test_safe_recovery_reuses_reserved_model_step_and_executes_once(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
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
    events = store.list_events("run-safe-recover")
    assert [event.kind for event in events].count(RunEventKind.MODEL_STEP_RESERVED) == 1
    assert [event.kind for event in events].count(RunEventKind.ACTIVATION_ABANDONED) == 1


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


@pytest.mark.asyncio
async def test_recovery_closes_multiple_claims_without_replaying_tools(
    tmp_path: Path,
) -> None:
    """第二个 claim 持久化后中断时，recovery 原子关闭全部 claim 且不重放。"""

    class CrashAfterSecondClaimStore(InMemoryLifecycleStore):
        claims = 0

        def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
            committed = super().claim_tool_call(command)
            self.claims += 1
            if self.claims == 2:
                raise IrisRunPersistenceError("second claim committed before crash")
            return committed

    effects: list[int] = []

    async def read_value(index: int) -> str:
        effects.append(index)
        return f"value-{index}"

    registry = ToolRegistry()
    registry.register_function(read_value, description="读取值")
    store = CrashAfterSecondClaimStore()
    first = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_batch_response(
                    ToolUseBlock(id="read-1", name="read_value", input={"index": 1}),
                    ToolUseBlock(id="read-2", name="read_value", input={"index": 2}),
                )
            ),
        ),
        store=store,
    )
    with pytest.raises(IrisRunPersistenceError, match="second claim committed"):
        await first.start(AgentRunRequest(input="执行", run_id="run-multi-claim-recover"))
    before_recovery_effects = list(effects)
    crashed = store.load_run("run-multi-claim-recover")
    assert crashed is not None and crashed.current_activation_id is not None
    assert all(
        record.phase is ToolCallPhase.CLAIMED
        for record in store.list_tool_calls("run-multi-claim-recover")
    )

    result = await AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry),
        store=store,
    ).recover(
        "run-multi-claim-recover",
        expected_activation_id=crashed.current_activation_id,
    )

    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert effects == before_recovery_effects
    records = store.list_tool_calls("run-multi-claim-recover")
    assert {record.tool_call_id for record in records} == {"read-1", "read-2"}
    assert all(record.phase is ToolCallPhase.OUTCOME_UNKNOWN for record in records)
    assert [event.kind for event in store.list_events("run-multi-claim-recover")][-4:] == [
        RunEventKind.ACTIVATION_ABANDONED,
        RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
        RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
        RunEventKind.RUN_TERMINAL,
    ]


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
async def test_active_recovery_requires_exact_activation_fence(tmp_path: Path) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    running = asyncio.create_task(runner.start(AgentRunRequest(input="恢复", run_id="run-fence")))
    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    second = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    with pytest.raises(IrisRunConflictError, match="expected_activation_id"):
        await second.recover("run-fence")
    with pytest.raises(IrisRunConflictError, match="fence"):
        await second.recover("run-fence", expected_activation_id="wrong")


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
async def test_recovery_rejects_environment_fingerprint_drift_without_mutation(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    first = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, system_text="原始配置"),
        store=store,
    )
    running = asyncio.create_task(
        first.start(AgentRunRequest(input="恢复", run_id="run-fingerprint-drift"))
    )
    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    before_run = store.load_run("run-fingerprint-drift")
    before_checkpoint = store.load_checkpoint("run-fingerprint-drift")
    before_events = store.list_events("run-fingerprint-drift")
    assert before_run is not None and before_run.current_activation_id is not None

    drifted = AgentRunner(
        runtime=build_runtime(tmp_path, system_text="已变更配置"),
        store=store,
    )
    with pytest.raises(IrisRunRecoveryError, match="fingerprint"):
        await drifted.recover(
            "run-fingerprint-drift",
            expected_activation_id=before_run.current_activation_id,
        )

    assert store.load_run("run-fingerprint-drift") == before_run
    assert store.load_checkpoint("run-fingerprint-drift") == before_checkpoint
    assert store.list_events("run-fingerprint-drift") == before_events
