"""AgentRunner start-to-terminal 测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisProviderError, IrisRunConflictError
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEvent,
    RunEventKind,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
    ToolErrorPolicy,
)
from iris.message import LLMRequest, LLMResponse, Msg, ToolUseBlock
from iris.runtime import SteeringInput
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    ToolCapability,
    ToolRegistry,
)

from .fakes import (
    BlockingProvider,
    CountingAgentRuntime,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


@pytest.mark.asyncio
async def test_runner_start_returns_reloaded_terminal_result(tmp_path: Path) -> None:
    """若 runner 返回 transient engine fact，durable result/history/events 会不一致。"""
    store = InMemoryLifecycleStore()
    runtime = build_runtime(tmp_path)
    runner = AgentRunner(runtime=runtime, store=store)

    result = await runner.start(
        AgentRunRequest(input="你好", session_id="session-1", run_id="run-1")
    )

    assert result == store.load_result("run-1")
    assert result.run.phase is RunPhase.TERMINAL
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.assistant_message is not None
    assert result.assistant_message.text == "完成"
    assert [message.role.value for message in store.load_session("session-1").messages] == [
        "user",
        "assistant",
    ]
    assert [event.kind for event in runner.list_events("run-1")] == [
        RunEventKind.RUN_STARTED,
        RunEventKind.ACTIVATION_STARTED,
        RunEventKind.MODEL_STEP_RESERVED,
        RunEventKind.MODEL_STEP_COMMITTED,
        RunEventKind.RUN_TERMINAL,
    ]
    assert runner.get_run("run-1") == result.run
    assert runner.get_result("run-1") == result


@pytest.mark.asyncio
async def test_start_activation_records_existing_raw_history_boundary(tmp_path: Path) -> None:
    """后续 run 的原始输入和历史起点取自创建事务。"""
    runtime = CountingAgentRuntime(
        build_runtime(tmp_path, provider=StaticProvider(text_response(), text_response()))
    )
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    await runner.start(AgentRunRequest(input="第一轮"))
    await runner.start(AgentRunRequest(input="第二轮"))

    assert [activation.run_input for activation in runtime.activations] == ["第一轮", "第二轮"]
    assert [activation.initial_session_message_count for activation in runtime.activations] == [
        0,
        2,
    ]


@pytest.mark.asyncio
async def test_managed_start_signals_after_registration_and_relays_live_events(
    tmp_path: Path,
) -> None:
    """Managed admission 不等待 provider，且 relay 覆盖 create、commit 与 finish。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    activation_started = asyncio.Event()
    relayed: list[RunEvent] = []
    running = asyncio.create_task(
        runner._start_managed(
            AgentRunRequest(input="等待", session_id="session-managed", run_id="run-managed"),
            durable_event_callback=relayed.append,
            activation_started=activation_started,
        )
    )

    try:
        await asyncio.wait_for(activation_started.wait(), timeout=1)
        active = store.load_run("run-managed")
        assert active is not None and active.phase is RunPhase.ACTIVE
        assert "run-managed" in runner._active
        assert not running.done()
        await asyncio.wait_for(provider.started.wait(), timeout=1)
    finally:
        provider.release.set()
    result = await running

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert relayed == store.list_events("run-managed")
    assert len({(event.run_id, event.sequence) for event in relayed}) == len(relayed)


@pytest.mark.asyncio
async def test_runner_cancellation_does_not_rescan_collected_event_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    relayed: list[RunEvent] = []
    running = asyncio.create_task(
        runner._start_managed(
            AgentRunRequest(input="等待", run_id="run-collector"),
            durable_event_callback=relayed.append,
        )
    )
    prior_sequence_reads = 0
    original_getattribute = RunEvent.__getattribute__

    try:
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        prior_ids = {id(event) for event in relayed}

        def getattribute(event: RunEvent, name: str) -> object:
            nonlocal prior_sequence_reads
            if name == "sequence" and id(event) in prior_ids:
                prior_sequence_reads += 1
            return original_getattribute(event, name)

        with monkeypatch.context() as patch:
            patch.setattr(RunEvent, "__getattribute__", getattribute)
            runner.request_cancel("run-collector")
    finally:
        provider.release.set()
    await running

    assert prior_sequence_reads == 0
    assert relayed == store.list_events("run-collector")


@pytest.mark.asyncio
async def test_managed_start_relays_model_commit_before_steering_ack(
    tmp_path: Path,
) -> None:
    """Durable RunEvent relay 必须先于同一 steering submission 的 delivered ack。"""
    order: list[str] = []

    class RecordingSteeringPort:
        def __init__(self) -> None:
            self.input: SteeringInput | None = SteeringInput(
                submission_id="submission-1",
                message=Msg.user("新方向"),
            )

        async def claim(self, run_id: str, activation_id: str) -> SteeringInput | None:
            del run_id, activation_id
            claimed, self.input = self.input, None
            return claimed

        def acknowledge(self, submission_id: str) -> None:
            assert submission_id == "submission-1"
            order.append("acknowledge")

        def fail(self, submission_id: str, reason: str) -> None:
            raise AssertionError(f"unexpected steering failure: {submission_id} {reason}")

    def record_event(event: RunEvent) -> None:
        if event.kind is RunEventKind.MODEL_STEP_COMMITTED:
            order.append("model-step-committed")

    provider = StaticProvider(text_response("第一轮"), text_response("最终轮"))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=InMemoryLifecycleStore(),
    )

    result = await runner._start_managed(
        AgentRunRequest(input="开始", run_id="run-steering-order"),
        steering=RecordingSteeringPort(),
        durable_event_callback=record_event,
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert order == ["model-step-committed", "acknowledge", "model-step-committed"]


@pytest.mark.asyncio
async def test_runner_maps_structured_engine_failure_to_durable_terminal(
    tmp_path: Path,
) -> None:
    """受控 provider 失败应作为结果返回，而不是逃逸异常。"""

    class FailingProvider:
        def estimate_input_tokens(self, request: LLMRequest) -> int:
            """为非计量测试返回固定输入估算。"""
            return 1

        async def complete(self, request: LLMRequest) -> LLMResponse:
            del request
            raise IrisProviderError("provider 不可用", provider="fake")

    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailingProvider()),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="失败", session_id="session-fail", run_id="run-fail")
    )

    assert result == store.load_result("run-fail")
    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert result.error.source == "provider"
    assert (
        sum(event.kind is RunEventKind.RUN_TERMINAL for event in store.list_events("run-fail")) == 1
    )


@pytest.mark.asyncio
async def test_runner_maps_tool_stop_policy_after_durable_error_result(
    tmp_path: Path,
) -> None:
    """STOP policy 也必须先提交工具错误事实再结算 run。"""
    provider = StaticProvider(tool_response(ToolUseBlock(id="missing-1", name="missing", input={})))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="调用缺失工具", run_id="run-tool-stop"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
        ),
    )

    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_ALLOWED"
    [tool_call] = store.list_tool_calls("run-tool-stop")
    assert tool_call.phase is ToolCallPhase.COMMITTED
    assert tool_call.result is not None and tool_call.result.is_error


@pytest.mark.asyncio
async def test_runner_returns_waiting_result_and_keeps_session_lane(tmp_path: Path) -> None:
    """waiting 必须已持久化 interaction，并继续占用 session lane。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"}))
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="写入", session_id="session-wait", run_id="run-wait")
    )

    assert result == store.load_result("run-wait")
    assert result.run.phase is RunPhase.WAITING
    assert result.pending_interaction is not None
    assert store.load_interaction(result.pending_interaction.interaction_id) is not None
    assert "run-wait" not in runner._active
    with pytest.raises(IrisRunConflictError, match="lane"):
        await runner.start(
            AgentRunRequest(
                input="冲突",
                session_id="session-wait",
                run_id="run-wait-conflict",
            )
        )
