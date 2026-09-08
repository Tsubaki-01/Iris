"""Harness live publisher 组合与 durable read facade 测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from iris.harness import AgentRunner, SessionManager, SubmissionEvent
from iris.harness.streaming import SessionSubmissionEvent
from iris.lifecycle import AgentRunRequest, RunEvent, RunPhase
from iris.message import ToolUseBlock
from iris.runtime import (
    AgentRuntime,
    RuntimeActivationInput,
    RuntimeActivationResult,
    RuntimeCommitPort,
    RuntimeEventSink,
    RuntimeSteeringPort,
    RuntimeStreamEvent,
)
from iris.store import InMemoryLifecycleStore
from iris.tools import CancellationSignal, ToolRegistry

from .fakes import (
    BlockingProvider,
    FailingPublisher,
    RecordingPublisher,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


class EmittingAgentRuntime(AgentRuntime):
    """在委托真实 runtime 前发出两条确定性 live fact。"""

    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime.environment)
        self.stream_sinks: list[RuntimeEventSink | None] = []

    async def execute(
        self,
        activation: RuntimeActivationInput,
        *,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None = None,
        stream_sink: RuntimeEventSink | None = None,
    ) -> RuntimeActivationResult:
        """记录 sink、发出 facts，再走非流式 provider 路径完成 run。"""
        self.stream_sinks.append(stream_sink)
        if stream_sink is not None:
            stream_sink.emit(
                RuntimeStreamEvent(
                    kind="model.step.started",
                    run_id=activation.run_id,
                    session_id=activation.session_id,
                    activation_id=activation.activation_id,
                    step_index=activation.cursor.step_index,
                )
            )
            stream_sink.emit(
                RuntimeStreamEvent(
                    kind="tool.preparing",
                    run_id=activation.run_id,
                    session_id=activation.session_id,
                    activation_id=activation.activation_id,
                    step_index=activation.cursor.step_index,
                    tool_call_id="tool-live",
                    tool_name="probe",
                    tool_ordinal=1,
                )
            )
        return await super().execute(
            activation,
            commits=commits,
            cancellation=cancellation,
            steering=steering,
        )


class RecordingObserver:
    """记录 settlement 后交付的 durable events。"""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def on_event(self, event: RunEvent) -> None:
        """记录一条 durable event。"""
        self.events.append(event)


async def _next_submission(
    stream: AsyncIterator[RunEvent | SubmissionEvent],
    submission_id: str,
) -> SubmissionEvent:
    """从 mixed stream 读取指定 submission event。"""
    while True:
        event = await asyncio.wait_for(anext(stream), timeout=1)
        if isinstance(event, SubmissionEvent) and event.submission_id == submission_id:
            return event


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_fails", [False, True])
async def test_runner_publishes_runtime_and_each_durable_event_once(
    tmp_path: Path,
    callback_fails: bool,
) -> None:
    """Runtime facts 保序，committed event 只经去重入口发布一次。"""
    publisher = RecordingPublisher()
    runtime = EmittingAgentRuntime(build_runtime(tmp_path))
    store = InMemoryLifecycleStore()
    observer = RecordingObserver()
    callback_events: list[RunEvent] = []

    def on_durable_event(event: RunEvent) -> None:
        callback_events.append(event)
        if callback_fails:
            raise RuntimeError("injected callback failure")

    runner = AgentRunner(
        runtime=runtime,
        store=store,
        observers=(observer,),
        live_publisher=publisher,
    )

    result = await runner._start_managed(
        AgentRunRequest(input="完成", run_id="run-publish", session_id="session-live"),
        durable_event_callback=on_durable_event,
    )

    runtime_facts = [fact for fact in publisher.facts if isinstance(fact, RuntimeStreamEvent)]
    durable_facts = [fact for fact in publisher.facts if isinstance(fact, RunEvent)]
    assert [fact.kind for fact in runtime_facts] == ["model.step.started", "tool.preparing"]
    assert runtime.stream_sinks[0] is not None
    assert durable_facts == store.list_events(result.run.run_id)
    assert callback_events == durable_facts
    assert observer.events == durable_facts
    assert len({(fact.run_id, fact.sequence) for fact in durable_facts}) == len(durable_facts)


@pytest.mark.asyncio
async def test_runner_publisher_failure_does_not_change_durable_result(
    tmp_path: Path,
) -> None:
    """Runtime 与 durable publish 失败都不能改变 settlement。"""
    publisher = FailingPublisher()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=EmittingAgentRuntime(build_runtime(tmp_path)),
        store=store,
        live_publisher=publisher,
    )

    result = await runner.start(
        AgentRunRequest(input="完成", run_id="run-failing-publisher", session_id="session-live")
    )

    assert result.run.phase is RunPhase.TERMINAL
    assert runner.get_result(result.run.run_id) == result
    assert store.list_events(result.run.run_id)
    assert any(isinstance(fact, RuntimeStreamEvent) for fact in publisher.facts)
    assert any(isinstance(fact, RunEvent) for fact in publisher.facts)


@pytest.mark.asyncio
async def test_manager_submission_side_channel_preserves_original_event(
    tmp_path: Path,
) -> None:
    """原 mixed stream 与 wrapper side channel 各接收一次 submission fact。"""
    provider = BlockingProvider()
    publisher = RecordingPublisher()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-submission",
        submission_publisher=publisher,
    )
    stream = manager.events()
    await manager.submit("开始")
    receipt = await manager.submit("下一轮", mode="follow_up")

    original = await _next_submission(stream, receipt.submission_id)
    wrapped = [
        fact
        for fact in publisher.facts
        if isinstance(fact, SessionSubmissionEvent)
        and fact.event.submission_id == receipt.submission_id
    ]

    assert original.state == "pending"
    assert wrapped == [SessionSubmissionEvent(session_id="session-submission", event=original)]
    provider.release.set()
    await stream.aclose()
    await manager.close()


@pytest.mark.asyncio
async def test_manager_publisher_failure_preserves_receipt_and_buffer(
    tmp_path: Path,
) -> None:
    """Submission publish 失败发生在原 buffer 写入之后。"""
    provider = BlockingProvider()
    publisher = FailingPublisher()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-failing-submission",
        submission_publisher=publisher,
    )
    stream = manager.events()
    await manager.submit("开始")

    receipt = await manager.submit("稍后处理", mode="follow_up")
    original = await _next_submission(stream, receipt.submission_id)

    assert receipt.state == "pending"
    assert original.state == "pending"
    assert (
        sum(
            isinstance(fact, SessionSubmissionEvent)
            and fact.event.submission_id == receipt.submission_id
            for fact in publisher.facts
        )
        == 1
    )
    provider.release.set()
    await stream.aclose()
    await manager.close()


@pytest.mark.asyncio
async def test_runner_durable_read_facades_delegate_exact_store(tmp_path: Path) -> None:
    """Session/tool reads 规范化 identity，并且没有 publisher side effect。"""
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="probe", description="探针")
    publisher = RecordingPublisher()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=EmittingAgentRuntime(
            build_runtime(
                tmp_path,
                registry=registry,
                provider=StaticProvider(
                    tool_response(ToolUseBlock(id="probe-1", name="probe", input={})),
                    text_response("完成"),
                ),
            )
        ),
        store=store,
        live_publisher=publisher,
    )
    result = await runner.start(
        AgentRunRequest(input="调用工具", run_id="run-read", session_id="session-read")
    )
    published_count = len(publisher.facts)

    session = runner.get_session("  session-read  ")
    calls = runner.list_tool_calls("  run-read  ")

    assert session == store.load_session("session-read")
    assert calls == store.list_tool_calls("run-read")
    assert len(calls) == 1
    assert len(publisher.facts) == published_count
    assert result.run.phase is RunPhase.TERMINAL
