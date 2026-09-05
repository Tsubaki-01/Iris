"""AgentRunner model-step budget 与 absolute deadline 测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from iris.exceptions import IrisProviderError
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEventKind,
    RunLimits,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import (
    LLMRequest,
    LLMResponse,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ProviderStreamError,
    ToolUseBlock,
)
from iris.runtime import (
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeCommitPort,
    RuntimeEventSink,
    RuntimeSteeringPort,
)
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    BaseTool,
    CancellationSignal,
    DefaultPermissionPolicy,
    PermissionDecision,
    ToolExecutionContext,
    ToolRegistry,
)

from .fakes import (
    FrozenClock,
    RecordingPublisher,
    StaticProvider,
    build_runtime,
    text_response,
    tool_batch_response,
    tool_response,
)


class DeadlineSignalRuntime:
    """在 engine 边界模拟 deadline signal 先于结果结算。"""

    def __init__(self, runtime: object, clock: FrozenClock) -> None:
        self.environment = cast(Any, runtime).environment
        self.clock = clock

    async def execute(
        self,
        activation: RuntimeActivationInput,
        *,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None = None,
        stream_sink: RuntimeEventSink | None = None,
    ) -> RuntimeActivationResult:
        del commits, steering, stream_sink
        self.clock.advance(seconds=2)
        cast(Any, cancellation).request_deadline()
        return RuntimeActivationResult(
            outcome=RuntimeActivationOutcome.CANCELLED,
            cursor=activation.cursor,
        )


@pytest.mark.asyncio
async def test_model_budget_is_reserved_before_provider_and_terminalized_once(
    tmp_path: Path,
) -> None:
    """第二个模型步没有预算时不得产生第二次 provider effect。"""
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="probe", description="探针")
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="probe-1", name="probe", input={})),
        text_response("不应调用"),
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="预算测试", run_id="run-budget"),
        options=AgentRunOptions(limits=RunLimits(max_model_steps=1)),
    )

    assert result.run.stop_reason is RunStopReason.BUDGET_EXHAUSTED
    assert len(provider.requests) == 1
    assert result.run.usage.model_steps_reserved == 1
    assert result.run.usage.model_steps_committed == 1
    assert (
        sum(event.kind is RunEventKind.RUN_TERMINAL for event in store.list_events("run-budget"))
        == 1
    )


@pytest.mark.asyncio
async def test_already_expired_start_skips_runtime_and_provider(tmp_path: Path) -> None:
    """已过期 deadline 应在 create transaction 内直接形成 durable terminal。"""
    clock = FrozenClock()
    provider = StaticProvider(text_response("不应调用"))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
        clock=clock,
    )

    result = await runner.start(
        AgentRunRequest(input="过期", run_id="run-expired"),
        options=AgentRunOptions(limits=RunLimits(deadline_at=clock.now() - timedelta(seconds=1))),
    )

    assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    assert provider.requests == []
    assert "run-expired" not in runner._active


@pytest.mark.asyncio
async def test_deadline_during_provider_wait_returns_deadline_terminal(
    tmp_path: Path,
) -> None:
    """run deadline 超时不能被误报为普通 provider timeout。"""

    class SlowProvider:
        def __init__(self) -> None:
            self.requests: list[LLMRequest] = []

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            await asyncio.sleep(10)
            return text_response("不应完成")

    provider = SlowProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
    )

    result = await runner.start(
        AgentRunRequest(input="等待 provider", run_id="run-provider-deadline"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(milliseconds=200))
        ),
    )

    assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("after_deadline", [False, True])
async def test_provider_cleanup_error_preserves_deadline_cause(
    tmp_path: Path,
    after_deadline: bool,
) -> None:
    """deadline 发起取消后 provider 清理抛错仍归因超时，到期前失败则保持原错误。"""

    class CleanupErrorProvider:
        """提供合法 complete 接口，模拟网络资源清理失败。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """按测试时序在请求阶段或取消清理阶段抛出 provider 错误。"""
            del request
            if after_deadline:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError as exc:
                    raise IrisProviderError("provider cleanup failed") from exc
            raise IrisProviderError("provider request failed")

    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=CleanupErrorProvider()),
        store=store,
    )
    result = await runner.start(
        AgentRunRequest(input="provider 清理", run_id="provider-cleanup"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(milliseconds=200))
        ),
    )

    assert result.run.stop_reason is (
        RunStopReason.DEADLINE_EXCEEDED if after_deadline else RunStopReason.FAILED
    )
    if after_deadline:
        assert result.error is None
    else:
        assert result.error is not None
        assert result.error.code == "PROVIDER_ERROR"
    assert store.load_session_lane("default") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_absolute_deadline_settles_provider_failure_before_timer_signal(
    tmp_path: Path,
    expired: bool,
    streaming: bool,
) -> None:
    """Clock 单调前进后先收到 provider 失败时，不依赖 timer 是否已经置位。"""
    clock = FrozenClock()
    entered = asyncio.Event()
    release = asyncio.Event()

    class EventFailureProvider:
        """通过事件同步合法 complete 异常与 typed stream terminal。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """在控制端推进时间后返回普通 provider 失败。"""
            del request
            entered.set()
            await release.wait()
            raise IrisProviderError("provider request failed")

        async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
            """按完整 started/failed 协议返回流式失败。"""
            del request
            scope = ModelStreamScope(
                model_stream_id="deadline-stream", provider="fake", model="fake-model", attempt=1
            )
            yield ModelResponseStarted(
                scope=scope, sequence=1, occurred_at=clock.now(), response_id="deadline-response"
            )
            entered.set()
            await release.wait()
            yield ModelResponseFailed(
                scope=scope,
                sequence=2,
                occurred_at=clock.now(),
                error=ProviderStreamError(
                    code="PROVIDER_STREAM_ERROR", message="provider stream failed", retryable=False
                ),
                semantic_output_emitted=False,
            )

    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=EventFailureProvider()),
        store=InMemoryLifecycleStore(),
        clock=clock,
        live_publisher=RecordingPublisher() if streaming else None,
    )
    running = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="provider 失败", run_id="unsignalled-deadline"),
            options=AgentRunOptions(
                limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=60))
            ),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    clock.advance(seconds=61 if expired else 1)
    assert not runner._active["unsignalled-deadline"].signal.deadline_requested
    release.set()
    result = await asyncio.wait_for(running, timeout=1)

    assert result.run.stop_reason is (
        RunStopReason.DEADLINE_EXCEEDED if expired else RunStopReason.FAILED
    )
    if expired:
        assert result.error is None
    else:
        assert result.error is not None
        assert result.error.code == ("PROVIDER_STREAM_ERROR" if streaming else "PROVIDER_ERROR")


@pytest.mark.asyncio
async def test_deadline_signal_before_effect_guard_prevents_new_tool_effect(
    tmp_path: Path,
) -> None:
    """最后一次 runtime 检查后到期，也不能越过 executor 的 pre-effect 检查。"""
    clock = FrozenClock()
    effects: list[str] = []

    class DeadlineOnExecutePolicy(DefaultPermissionPolicy):
        def __init__(self) -> None:
            super().__init__(write_mode="allow")
            self.checks = 0

        def check(
            self,
            tool: BaseTool,
            params: dict[str, Any],
            context: ToolExecutionContext,
        ) -> PermissionDecision:
            self.checks += 1
            if self.checks == 2:
                assert context.cancellation is not None
                clock.advance(seconds=2)
                cast(Any, context.cancellation).request_deadline()
            return super().check(tool, params, context)

    def effect() -> str:
        effects.append("effect")
        return "effect"

    registry = ToolRegistry()
    registry.register_function(effect, description="副作用")
    store = InMemoryLifecycleStore()
    policy = DeadlineOnExecutePolicy()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            permission_policy=policy,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="deadline-effect-1", name="effect", input={}))
            ),
        ),
        store=store,
        clock=clock,
    )

    result = await runner.start(
        AgentRunRequest(input="deadline effect", run_id="run-deadline-effect"),
        options=AgentRunOptions(limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=1))),
    )

    assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    assert effects == []
    assert policy.checks == 2
    [tool_call] = store.list_tool_calls("run-deadline-effect")
    assert tool_call.phase is ToolCallPhase.PREPARED
    [tool_result] = store.load_session("default").messages[-1].tool_results
    assert tool_result.tool_use_id == "deadline-effect-1"
    assert tool_result.metadata["error"]["code"] == "TOOL_NOT_STARTED"


@pytest.mark.asyncio
async def test_provider_timeout_without_expired_deadline_remains_failure(
    tmp_path: Path,
) -> None:
    """operation 自身 TimeoutError 不能仅因存在未过期 run deadline 就改写。"""

    class TimeoutProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            del request
            raise TimeoutError("provider operation timeout")

    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=TimeoutProvider()),
        store=InMemoryLifecycleStore(),
    )

    result = await runner.start(
        AgentRunRequest(input="provider timeout", run_id="run-provider-timeout"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=5))
        ),
    )

    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert result.error.code == "PROVIDER_TIMEOUT"


@pytest.mark.asyncio
async def test_claimed_tool_deadline_maps_to_outcome_unknown(tmp_path: Path) -> None:
    """effect 已 claim 后超时无法证明是否执行，必须 fail closed。"""

    async def slow_effect() -> str:
        await asyncio.sleep(10)
        return "不应完成"

    registry = ToolRegistry()
    registry.register_function(slow_effect, description="慢副作用")
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="slow-1", name="slow_effect", input={}))
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="慢工具", run_id="run-tool-deadline"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(milliseconds=200))
        ),
    )

    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert result.error is not None
    assert result.error.code == "TOOL_OUTCOME_UNKNOWN"
    [tool_call] = store.list_tool_calls("run-tool-deadline")
    assert tool_call.phase is ToolCallPhase.OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_parallel_claims_deadline_atomically_settles_outcome_unknown(
    tmp_path: Path,
) -> None:
    """deadline 中断并发 claimed bodies 时必须关闭全部 unresolved claims。"""
    started: set[int] = set()

    async def slow_read(index: int) -> str:
        started.add(index)
        await asyncio.sleep(10)
        return "不应完成"

    registry = ToolRegistry()
    registry.register_function(slow_read, description="慢读取")
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_batch_response(
                    ToolUseBlock(id="slow-1", name="slow_read", input={"index": 1}),
                    ToolUseBlock(id="slow-2", name="slow_read", input={"index": 2}),
                )
            ),
        ),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="并发 deadline", run_id="run-parallel-deadline"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(milliseconds=200))
        ),
    )

    assert started == {1, 2}
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    records = store.list_tool_calls("run-parallel-deadline")
    assert {record.tool_call_id for record in records} == {"slow-1", "slow-2"}
    assert all(record.phase is ToolCallPhase.OUTCOME_UNKNOWN for record in records)
