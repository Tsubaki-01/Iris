"""命令环境先停止、run 后结算，并可只重试失败的清理。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisExecutionCleanupError,
    IrisProviderError,
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.execution import (
    CommandEnvironment,
    CommandOutcome,
    CommandRequest,
    ExecutionBinding,
    ExecutionConfig,
    ExecutionMode,
    ExecutionScope,
    ExecutionStopReceipt,
)
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CheckpointResumability,
    ClaimToolCall,
    RunCommit,
    RunEvent,
    RunEventKind,
    RunLimits,
    RunPhase,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import (
    BlockingProvider,
    RecordingPublisher,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


class ControlledStop:
    """物理停止与排空分别由测试推进。"""

    def __init__(self, service: ControlledService) -> None:
        self.service = service

    async def wait_stopped(self) -> ExecutionStopReceipt:
        """提供当前停止的物理完成点。"""
        await self.service.stopped.wait()
        return self.service.receipt

    async def wait_drained(self) -> ExecutionStopReceipt:
        """失败不声称物理停止或排空成功。"""
        if self.service.fail:
            if self.service.fail_release is not None:
                await self.service.fail_release.wait()
            raise IrisExecutionCleanupError("stop failed")
        await self.wait_stopped()
        await self.service.drained.wait()
        return self.service.receipt


class ControlledService:
    """不启动进程，只观测 runner 的停止时序。"""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.stopped = asyncio.Event()
        self.drained = asyncio.Event()
        self.receipt = ExecutionStopReceipt("service", "stop")
        self.calls: list[ExecutionScope] = []
        self.fail = False
        self.close_fail = False
        self.close_calls = 0
        self.fail_release: asyncio.Event | None = None

    async def prepare(self) -> None:
        """无外部资源。"""

    async def execute(self, scope: ExecutionScope, request: CommandRequest) -> CommandOutcome:
        """本组不从命令入口制造模型错误。"""
        raise AssertionError("no command expected")

    def stop(self, scope: ExecutionScope) -> ControlledStop:
        """同步记录停止准入。"""
        self.calls.append(scope)
        self.entered.set()
        return ControlledStop(self)

    async def wait_drained(self, receipt: ExecutionStopReceipt) -> None:
        """消费已有收据不登记另一轮停止。"""
        assert receipt is self.receipt
        await self.drained.wait()

    async def aclose(self) -> None:
        """模拟第一次关闭失败、后续可重试。"""
        self.close_calls += 1
        if self.close_fail:
            raise IrisExecutionCleanupError("close failed")

    def release(self) -> None:
        """使两个完成点都可观察。"""
        self.stopped.set()
        self.drained.set()


class FailedProvider:
    """确定失败，不依赖远程模型。"""

    def __init__(self) -> None:
        self.calls = 0

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """固定估算。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """标记调用次数后失败。"""
        self.calls += 1
        raise IrisProviderError("original provider failure")


def bind_service(runner: AgentRunner, service: ControlledService) -> None:
    """给已有真实 runtime 注入可观测命令资源。"""
    runner.runtime.environment.execution_binding = ExecutionBinding(
        ExecutionConfig(),
        service,
        CommandEnvironment("Linux", ExecutionMode.NATIVE, "Linux", "/bin/sh"),
    )


@pytest.mark.asyncio
async def test_failure_keeps_lane_until_stop_and_drain(tmp_path: Path) -> None:
    provider = FailedProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    bind_service(runner, service)
    task = asyncio.create_task(runner.start(AgentRunRequest(input="x", run_id="failed")))
    await asyncio.wait_for(service.entered.wait(), 1)
    assert runner.get_run("failed").phase is RunPhase.ACTIVE
    assert runner.get_result("failed") is None
    service.stopped.set()
    await asyncio.sleep(0)
    assert not task.done()
    assert runner.get_run("failed").phase is RunPhase.ACTIVE
    service.drained.set()
    result = await task
    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error.code == "PROVIDER_ERROR"
    assert provider.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["cancel", "recover", "resume"])
async def test_cleanup_retry_preserves_original_result(tmp_path: Path, retry: str) -> None:
    provider = FailedProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    service.fail = True
    bind_service(runner, service)
    with pytest.raises(IrisExecutionCleanupError):
        await runner.start(AgentRunRequest(input="x", run_id="failed"))
    assert runner.get_run("failed").phase is RunPhase.ACTIVE
    service.fail = False
    service.release()
    if retry == "cancel":
        result = await runner.cancel("failed")
    elif retry == "recover":
        result = await runner.recover("failed")
    else:
        from iris.hitl import PermissionInteractionResponse

        result = await runner.resume(
            "failed",
            interaction_id="unused",
            response=PermissionInteractionResponse(decision="approve"),
        )
    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error.code == "PROVIDER_ERROR"
    assert result.run.cancellation_requested_at is None
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_expired_start_stops_before_input_or_terminal(tmp_path: Path) -> None:
    provider = StaticProvider(text_response("unused"))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    bind_service(runner, service)
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="not committed", run_id="expired"),
            options=AgentRunOptions(
                limits=RunLimits(deadline_at=datetime.now(UTC) - timedelta(seconds=1))
            ),
        )
    )
    await asyncio.wait_for(service.entered.wait(), 1)
    assert runner.get_run("expired").phase is RunPhase.ACTIVE
    assert runner.get_session("default").messages == []
    assert provider.requests == []
    service.release()
    assert (await task).run.stop_reason is RunStopReason.DEADLINE_EXCEEDED


@pytest.mark.asyncio
async def test_waiting_deadline_settles_without_new_input(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_function(
        lambda: "x", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    provider = StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry),
        store=InMemoryLifecycleStore(),
    )
    service = ControlledService()
    bind_service(runner, service)
    waiting = await runner.start(
        AgentRunRequest(input="wait", run_id="waiting"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(milliseconds=80))
        ),
    )
    assert waiting.run.phase is RunPhase.WAITING
    await asyncio.wait_for(service.entered.wait(), 1)
    assert runner.get_run("waiting").phase is RunPhase.WAITING
    service.release()
    for _ in range(100):
        if runner.get_run("waiting").phase is RunPhase.TERMINAL:
            break
        await asyncio.sleep(0.01)
    assert runner.get_result("waiting").run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    await runner.aclose()


@pytest.mark.asyncio
async def test_raw_task_cancel_cleans_without_durable_cancel(tmp_path: Path) -> None:
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    bind_service(runner, service)
    task = asyncio.create_task(runner.start(AgentRunRequest(input="x", run_id="raw")))
    await provider.started.wait()
    task.cancel()
    await asyncio.wait_for(service.entered.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    service.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.get_run("raw").phase is RunPhase.ACTIVE
    assert runner.get_run("raw").cancellation_requested_at is None


@pytest.mark.asyncio
async def test_close_retries_owned_resources_after_failure(tmp_path: Path) -> None:
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    service = ControlledService()
    bind_service(runner, service)
    service.close_fail = True
    with pytest.raises(IrisExecutionCleanupError):
        await runner.aclose()
    service.close_fail = False
    await runner.aclose()
    await runner.aclose()
    assert service.close_calls == 2


@pytest.mark.asyncio
async def test_pending_waiters_share_one_attempt_and_cancel_is_local(tmp_path: Path) -> None:
    provider = FailedProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    bind_service(runner, service)
    first = asyncio.create_task(runner.start(AgentRunRequest(input="x", run_id="shared")))
    await service.entered.wait()
    recover = asyncio.create_task(runner.recover("shared"))
    cancel = asyncio.create_task(runner.cancel("shared"))
    await asyncio.sleep(0)
    cancel.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancel
    assert not first.done() and not recover.done()
    assert len(service.calls) == 1
    service.release()
    result = await first
    assert await recover == result
    assert result.run.stop_reason is RunStopReason.FAILED
    assert (
        sum(event.kind is RunEventKind.RUN_TERMINAL for event in runner.store.list_events("shared"))
        == 1
    )


@pytest.mark.asyncio
async def test_failed_attempt_reports_once_to_all_waiters(tmp_path: Path) -> None:
    from iris.harness import ExecutionCleanupFailed

    publisher = RecordingPublisher()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailedProvider()),
        store=InMemoryLifecycleStore(),
        live_publisher=publisher,
    )
    service = ControlledService()
    service.fail = True
    service.fail_release = asyncio.Event()
    bind_service(runner, service)
    first = asyncio.create_task(runner.start(AgentRunRequest(input="x", run_id="failed")))
    await service.entered.wait()
    other = asyncio.create_task(runner.cancel("failed"))
    await asyncio.sleep(0)
    service.fail_release.set()
    results = await asyncio.gather(first, other, return_exceptions=True)
    assert all(isinstance(item, IrisExecutionCleanupError) for item in results), results
    assert len([fact for fact in publisher.facts if isinstance(fact, ExecutionCleanupFailed)]) == 1
    assert len(service.calls) == 1
    service.fail = False
    service.release()
    await runner.recover("failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlite", [False, True])
async def test_unknown_recovery_owns_new_fence_before_stop(tmp_path: Path, sqlite: bool) -> None:
    store = SQLiteStore(tmp_path / "lifecycle.db") if sqlite else InMemoryLifecycleStore()
    registry = ToolRegistry()
    registry.register_function(lambda: "must not execute", name="effect", description="副作用")
    provider = StaticProvider(tool_response(ToolUseBlock(id="effect", name="effect", input={})))
    first = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry), store=store
    )
    original_claim = store.claim_tool_call

    def crash(command: ClaimToolCall) -> RunCommit:
        original_claim(command)
        raise IrisRunPersistenceError("crash after claim")

    store.claim_tool_call = crash
    with pytest.raises(IrisRunPersistenceError):
        await first.start(AgentRunRequest(input="x", run_id="unknown"))
    store.claim_tool_call = original_claim
    old = store.load_run("unknown")
    runner = AgentRunner(runtime=build_runtime(tmp_path, registry=registry), store=store)
    service = ControlledService()
    bind_service(runner, service)
    task = asyncio.create_task(
        runner.recover("unknown", expected_activation_id=old.current_activation_id)
    )
    await asyncio.wait_for(service.entered.wait(), 1)
    current = store.load_run("unknown")
    assert current.phase is RunPhase.ACTIVE
    assert current.current_activation_id != old.current_activation_id
    assert store.load_checkpoint("unknown").resumability is CheckpointResumability.BLOCKED_UNKNOWN
    assert store.list_tool_calls("unknown")[0].phase is ToolCallPhase.CLAIMED
    loser = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    loser_service = ControlledService()
    bind_service(loser, loser_service)
    with pytest.raises(IrisRunConflictError):
        await loser.recover("unknown", expected_activation_id=old.current_activation_id)
    assert loser_service.calls == []
    service.release()
    assert (await task).run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert store.list_tool_calls("unknown")[0].phase is ToolCallPhase.OUTCOME_UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlite", [False, True])
async def test_budget_refusal_keeps_active_until_stop(tmp_path: Path, sqlite: bool) -> None:
    store = SQLiteStore(tmp_path / "lifecycle.db") if sqlite else InMemoryLifecycleStore()
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="probe", description="读取")
    provider = StaticProvider(tool_response(ToolUseBlock(id="probe", name="probe", input={})))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry), store=store
    )
    service = ControlledService()
    bind_service(runner, service)
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="x", run_id="budget"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=1)),
        )
    )
    await asyncio.wait_for(service.entered.wait(), 1)
    assert runner.get_run("budget").phase is RunPhase.ACTIVE
    assert runner.get_run("budget").usage.model_steps_reserved == 1
    assert len(provider.requests) == 1
    service.release()
    assert (await task).run.stop_reason is RunStopReason.BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_close_retries_pending_and_rejects_new_business(tmp_path: Path) -> None:
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailedProvider()), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    service.fail = True
    bind_service(runner, service)
    with pytest.raises(IrisExecutionCleanupError):
        await runner.start(AgentRunRequest(input="x", run_id="failed"))
    with pytest.raises(IrisExecutionCleanupError):
        await runner.aclose()
    with pytest.raises(IrisRunStateError, match="关闭"):
        await runner.start(AgentRunRequest(input="x", session_id="another"))
    assert service.close_calls == 0
    service.fail = False
    service.release()
    await runner.aclose()
    assert runner.get_result("failed").run.stop_reason is RunStopReason.FAILED
    assert service.close_calls == 1


@pytest.mark.asyncio
async def test_close_reuses_pending_completed_by_another_waiter(tmp_path: Path) -> None:
    """关闭快照中的后一项完成后，不能再发停止或第二次 FinishRun。"""

    class PerRunService(ControlledService):
        def __init__(self) -> None:
            super().__init__()
            self.scopes = {name: ControlledService() for name in ("a", "b")}
            for item in self.scopes.values():
                item.fail = True

        def stop(self, scope: ExecutionScope) -> ControlledStop:
            return self.scopes[scope.run_id].stop(scope)

    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailedProvider()), store=InMemoryLifecycleStore()
    )
    service = PerRunService()
    bind_service(runner, service)
    for name in ("a", "b"):
        with pytest.raises(IrisExecutionCleanupError):
            await runner.start(AgentRunRequest(input="x", run_id=name, session_id=name))
        service.scopes[name].fail = False
        service.scopes[name].entered.clear()
    closing = asyncio.create_task(runner.aclose())
    await service.scopes["a"].entered.wait()
    # close 已封锁新业务；已加入的独立 cleanup waiter 仍可推进。
    pending_b = runner._execution_lifecycle.pending["b"]
    finishing_b = asyncio.create_task(runner._execution_lifecycle.join(pending_b))
    await service.scopes["b"].entered.wait()
    service.scopes["b"].release()
    await finishing_b
    service.scopes["a"].release()
    await closing
    assert len(service.scopes["b"].calls) == 2
    assert service.close_calls == 1


@pytest.mark.asyncio
async def test_cleanup_retry_delivers_each_durable_event_once(tmp_path: Path) -> None:
    """第一次失败的前缀与重试后的终态分别投递，不能重复旧事件。"""
    delivered: list[RunEvent] = []

    class Observer:
        """记录异步 observer 的实际投递。"""

        async def on_event(self, event: RunEvent) -> None:
            """保存一次投递。"""
            delivered.append(event)

    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailedProvider()),
        store=InMemoryLifecycleStore(),
        observers=(Observer(),),
    )
    service = ControlledService()
    service.fail = True
    bind_service(runner, service)
    with pytest.raises(IrisExecutionCleanupError):
        await runner.start(AgentRunRequest(input="x", run_id="failed"))
    service.fail = False
    service.release()
    await runner.recover("failed")
    assert delivered == runner.store.list_events("failed")


@pytest.mark.asyncio
async def test_waiting_cancel_delivers_request_once_to_observer_and_live(tmp_path: Path) -> None:
    delivered: list[RunEvent] = []

    class Observer:
        """检查等待取消的异步事件序列。"""

        async def on_event(self, event: RunEvent) -> None:
            """保留原始事件。"""
            delivered.append(event)

    registry = ToolRegistry()
    registry.register_function(
        lambda: "ok", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    publisher = RecordingPublisher()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write", name="write", input={}))
            ),
        ),
        store=InMemoryLifecycleStore(),
        observers=(Observer(),),
        live_publisher=publisher,
    )
    await runner.start(AgentRunRequest(input="x", run_id="waiting"))
    await runner.cancel("waiting")
    expected = runner.store.list_events("waiting")
    assert delivered == expected
    assert [fact for fact in publisher.facts if isinstance(fact, RunEvent)] == expected


@pytest.mark.asyncio
async def test_cleanup_only_retry_receipt_reaches_same_unknown_recovery(tmp_path: Path) -> None:
    class CancelAfterClaimStore(InMemoryLifecycleStore):
        """在claim落库后模拟裸取消，工具尚未开始。"""

        def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
            """已有claim必须由后续UNKNOWN恢复关闭。"""
            super().claim_tool_call(command)
            raise asyncio.CancelledError

    class RetryThenRestartService(ControlledService):
        """重试停止完成后即允许其他session开始，不接受旧链第二次stop。"""

        def stop(self, scope: ExecutionScope) -> ControlledStop:
            if len(self.calls) >= 2:
                raise AssertionError("old cleanup stopped restarted environment")
            return super().stop(scope)

    registry = ToolRegistry()
    registry.register_function(lambda: "must not run", name="effect", description="副作用")
    store = CancelAfterClaimStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="call", name="effect", input={}))
            ),
        ),
        store=store,
    )
    service = RetryThenRestartService()
    service.fail = True
    bind_service(runner, service)
    with pytest.raises(IrisExecutionCleanupError):
        await runner.start(AgentRunRequest(input="x", run_id="unknown"))
    old = store.load_run("unknown")
    assert old.phase is RunPhase.ACTIVE
    assert runner._execution_lifecycle.pending["unknown"].stop_reason is None
    service.fail = False
    service.release()
    result = await runner.recover("unknown", expected_activation_id=old.current_activation_id)
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert len(service.calls) == 2
