"""AgentRunner model-step budget 与 absolute deadline 测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEventKind,
    RunLimits,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.runtime import (
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeCommitPort,
)
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    BaseTool,
    CancellationSignal,
    DefaultPermissionPolicy,
    PermissionDecision,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)

from .fakes import FrozenClock, StaticProvider, build_runtime, text_response, tool_response


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
    ) -> RuntimeActivationResult:
        del commits
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
async def test_deadline_signal_cannot_be_durable_cancelled(tmp_path: Path) -> None:
    """timer 与 engine 边界竞态时，deadline 原因不能退化成 caller cancel。"""
    clock = FrozenClock()
    runtime = DeadlineSignalRuntime(build_runtime(tmp_path), clock)
    runner = AgentRunner(
        runtime=cast(Any, runtime),
        store=InMemoryLifecycleStore(),
        clock=clock,
    )

    result = await runner.start(
        AgentRunRequest(input="deadline signal", run_id="run-deadline-signal"),
        options=AgentRunOptions(limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=1))),
    )

    assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED


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
            if self.checks == 3:
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
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            permission_policy=DeadlineOnExecutePolicy(),
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
    [tool_call] = store.list_tool_calls("run-deadline-effect")
    assert tool_call.phase is ToolCallPhase.PREPARED


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
    assert tool_call.phase is ToolCallPhase.CLAIMED


@pytest.mark.asyncio
async def test_claimed_tool_cancellation_maps_to_outcome_unknown(tmp_path: Path) -> None:
    """claim 之后观察到 cooperative cancellation 时也不能假定 effect 未发生。"""

    class CancellingTool(BaseTool):
        definition = ToolDefinition(
            name="cancel_after_claim",
            description="claim 后请求取消",
            input_schema={"type": "object", "properties": {}},
        )

        async def arun(
            self,
            params: BaseModel | dict[str, Any],
            context: ToolExecutionContext,
        ) -> ToolResult:
            del params
            assert context.cancellation is not None
            cast(Any, context.cancellation).request()
            context.cancellation.raise_if_requested()
            raise AssertionError("取消后不应继续")

    registry = ToolRegistry()
    registry.register(CancellingTool())
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(
                        id="cancel-1",
                        name="cancel_after_claim",
                        input={},
                    )
                )
            ),
        ),
        store=store,
    )

    result = await runner.start(AgentRunRequest(input="取消工具", run_id="run-tool-cancel"))

    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert result.error is not None
    assert result.error.code == "TOOL_OUTCOME_UNKNOWN"
    [tool_call] = store.list_tool_calls("run-tool-cancel")
    assert tool_call.phase is ToolCallPhase.CLAIMED
