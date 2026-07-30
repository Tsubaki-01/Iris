"""AgentRunner activation-scope live resource settlement 测试。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.exceptions import IrisProviderError, IrisRunPersistenceError
from iris.harness import AgentRunner
from iris.harness.runner import ActiveActivation
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    ClaimToolCall,
    FinishRun,
    RunCommit,
    RunEventKind,
    RunLimits,
    RunPhase,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import StaticProvider, build_runtime, text_response, tool_response


class TrackingRunner(AgentRunner):
    """保留最近注册的 activation 供 finally 后断言。"""

    registered: ActiveActivation | None = None

    def _register(
        self,
        active: ActiveActivation,
        current_activation_id: str | None,
    ) -> None:
        super()._register(active, current_activation_id)
        self.registered = active


def _assert_settled(runner: TrackingRunner, run_id: str) -> None:
    assert run_id not in runner._active
    assert runner.registered is not None
    assert runner.registered.task is None
    assert runner.registered.deadline_task is None


@pytest.mark.asyncio
async def test_success_settles_activation_resources_without_closing_environment(
    tmp_path: Path,
) -> None:
    """一次 run 只清理 activation 资源，不关闭 caller-owned provider/store。"""

    class EnvironmentProvider(StaticProvider):
        def __init__(self) -> None:
            super().__init__(text_response())
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class EnvironmentStore(InMemoryLifecycleStore):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    provider = EnvironmentProvider()
    store = EnvironmentStore()
    runner = TrackingRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="完成", run_id="run-success"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=5))
        ),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    _assert_settled(runner, "run-success")
    assert provider.close_calls == 0
    assert store.close_calls == 0


@pytest.mark.asyncio
async def test_waiting_and_structured_failure_both_settle_live_resources(
    tmp_path: Path,
) -> None:
    """waiting/failed durable 结果都不能遗留 process-local activation。"""

    def write() -> str:
        return "write"

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    waiting_runner = TrackingRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write-1", name="write", input={}))
            ),
        ),
        store=InMemoryLifecycleStore(),
    )
    waiting = await waiting_runner.start(
        AgentRunRequest(input="等待", session_id="wait", run_id="run-waiting")
    )

    class FailingProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            del request
            raise IrisProviderError("失败", provider="fake")

    failed_runner = TrackingRunner(
        runtime=build_runtime(tmp_path, provider=FailingProvider()),
        store=InMemoryLifecycleStore(),
    )
    failed = await failed_runner.start(
        AgentRunRequest(input="失败", session_id="fail", run_id="run-failed")
    )

    assert waiting.run.phase is RunPhase.WAITING
    assert failed.run.stop_reason is RunStopReason.FAILED
    _assert_settled(waiting_runner, "run-waiting")
    _assert_settled(failed_runner, "run-failed")


@pytest.mark.asyncio
async def test_persistence_exception_propagates_after_live_resource_settlement(
    tmp_path: Path,
) -> None:
    """finish commit 失败时不能伪造结果，但 finally 仍须清理 live 资源。"""

    class FailingFinishStore(InMemoryLifecycleStore):
        def finish_run(self, command: FinishRun) -> RunCommit:
            del command
            raise IrisRunPersistenceError("模拟 finish persistence failure")

    store = FailingFinishStore()
    runner = TrackingRunner(runtime=build_runtime(tmp_path), store=store)

    with pytest.raises(IrisRunPersistenceError, match="finish persistence"):
        await runner.start(AgentRunRequest(input="持久化失败", run_id="run-persistence-failure"))

    _assert_settled(runner, "run-persistence-failure")
    stored = store.load_run("run-persistence-failure")
    assert stored is not None
    assert stored.phase is RunPhase.ACTIVE
    assert store.load_result("run-persistence-failure") is None


@pytest.mark.asyncio
async def test_unexpected_exception_after_claim_is_outcome_unknown(tmp_path: Path) -> None:
    """claim 已提交后发生意外异常时，runner 不能证明工具 effect 未发生。"""

    class CrashAfterClaimStore(InMemoryLifecycleStore):
        def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
            super().claim_tool_call(command)
            raise RuntimeError("claim committed before engine crash")

    registry = ToolRegistry()
    registry.register_function(lambda: "effect", name="effect", description="副作用")
    store = CrashAfterClaimStore()
    runner = TrackingRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="effect-1", name="effect", input={}))
            ),
        ),
        store=store,
    )

    result = await runner.start(AgentRunRequest(input="执行", run_id="run-crash-after-claim"))

    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert result.error is not None
    assert result.error.code == "TOOL_OUTCOME_UNKNOWN"
    [record] = store.list_tool_calls("run-crash-after-claim")
    assert record.phase is ToolCallPhase.OUTCOME_UNKNOWN
    assert RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN in {
        event.kind for event in store.list_events("run-crash-after-claim")
    }
    _assert_settled(runner, "run-crash-after-claim")
