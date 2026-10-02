"""Run Hooks 的真实 Runner 开始、终态与驱动取消边界。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from iris.command import CommandStopSlot
from iris.exceptions import (
    IrisCommandCleanupError,
    IrisRunPersistenceError,
    IrisRunStateError,
    IrisToolOutcomeUnknownError,
)
from iris.harness import AgentRunner
from iris.harness.streaming import CommandCleanupFailed
from iris.hitl import QuestionInteractionResponse
from iris.hooks import HookEvent, HookRegistration, RunFinishedEvent, RunStartedEvent
from iris.hooks._dispatch_types import CommandHookRegistration, HookControl, HookInvocationOutcome
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    FinishRun,
    RunCommit,
    RunEvent,
    RunLimits,
    RunPhase,
    RunStopReason,
)
from iris.message import DataBlock, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.store import InMemoryLifecycleStore
from iris.tools import AskQuestionTool, CancellationSignal, ToolRegistry

from .fakes import (
    FrozenClock,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)
from .test_command_settlement import ControlledService, bind_service


def _runner(
    tmp_path: Path,
    *registrations: HookRegistration | CommandHookRegistration,
    store: InMemoryLifecycleStore | None = None,
    provider: StaticProvider | None = None,
) -> AgentRunner:
    runtime = build_runtime(tmp_path, provider=provider)
    runtime = AgentRuntime(
        replace(runtime.environment, hook_dispatcher=HookDispatcher(registrations))
    )
    return AgentRunner(runtime=runtime, store=store or InMemoryLifecycleStore())


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["string", "mixed", "image-only"])
async def test_started_precedes_model_and_finished_follows_unique_commit(
    tmp_path: Path, input_kind: str
) -> None:
    provider = StaticProvider(text_response())
    observed: list[HookEvent] = []
    observations: list[bool] = []

    async def record(event: HookEvent) -> None:
        observed.append(event)
        if isinstance(event, RunStartedEvent):
            observations.extend(
                [
                    not provider.requests,
                    event.input == content,
                    event.run.phase is RunPhase.ACTIVE,
                    event.activation_id in {item.activation_id for item in runner._active.values()},
                ]
            )
        elif isinstance(event, RunFinishedEvent):
            observations.extend(
                [
                    runner.get_result("run") == event.result,
                    "run" not in runner._command_lifecycle.pending,
                ]
            )

    runner = _runner(
        tmp_path,
        HookRegistration(event="run.started", name="start", handler=record),
        HookRegistration(event="run.finished", name="finish", handler=record),
        provider=provider,
    )
    content: str | list[DataBlock] = "input"
    if input_kind != "string":
        source = tmp_path / "原图.png"
        with Image.new("RGB", (8, 5), "red") as image:
            image.save(source)
        imported = await runner.import_image(source, session_id="default", name="原图.png")
        content = [imported] if input_kind == "image-only" else [TextBlock(text="input"), imported]
    result = await runner.start(AgentRunRequest(input=content, run_id="run"))
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert [event.event for event in observed] == ["run.started", "run.finished"]
    assert observations == [True] * 6
    assert provider.requests[0].messages[1].content == content
    assert await runner.recover("run", expected_activation_id="unused") == result
    assert len(observed) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["run", "deadline", "sdk", "unknown"])
async def test_started_control_stops_before_first_model(tmp_path: Path, control: str) -> None:
    entered = asyncio.Event()
    provider = StaticProvider()

    async def before(event: HookEvent) -> None:
        entered.set()
        if control == "unknown":
            raise IrisToolOutcomeUnknownError("hook effect unknown")
        await asyncio.Event().wait()

    runner = _runner(
        tmp_path,
        HookRegistration(event="run.started", name="start", handler=before),
        provider=provider,
    )
    options = (
        AgentRunOptions(limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=0.1)))
        if control == "deadline"
        else None
    )
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="input", run_id="run"), options=options)
    )
    await entered.wait()
    if control == "run":
        runner.request_cancel("run")
        result = await task
        assert result.run.stop_reason is RunStopReason.CANCELLED
    elif control == "deadline":
        result = await task
        assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    elif control == "sdk":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runner.get_run("run").phase is RunPhase.ACTIVE
        assert not runner._command_lifecycle.pending
    else:
        result = await task
        assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
        assert result.error is not None
        assert result.error.code == "TOOL_OUTCOME_UNKNOWN"
        assert result.error.source == "tool"
    assert not provider.requests


@pytest.mark.asyncio
async def test_waiting_and_resume_do_not_repeat_started(tmp_path: Path) -> None:
    events: list[str] = []

    async def record(event: HookEvent) -> None:
        events.append(event.event)

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    runtime = build_runtime(
        tmp_path,
        registry=registry,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="q", name="ask_question", input={"question": "继续？"})),
            text_response(),
        ),
    )
    runtime = AgentRuntime(
        replace(
            runtime.environment,
            hook_dispatcher=HookDispatcher(
                [
                    HookRegistration(event="run.started", name="start", handler=record),
                    HookRegistration(event="run.finished", name="finish", handler=record),
                ]
            ),
        )
    )
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    waiting = await runner.start(AgentRunRequest(input="input", run_id="run"))
    assert waiting.pending_interaction is not None
    assert events == ["run.started"]
    result = await runner.resume(
        "run",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="继续"),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert events == ["run.started", "run.finished"]


@pytest.mark.asyncio
async def test_started_ordinary_error_continues(tmp_path: Path) -> None:
    async def fail(event: HookEvent) -> None:
        raise RuntimeError("ordinary hook failure")

    runner = _runner(tmp_path, HookRegistration(event="run.started", name="start", handler=fail))
    result = await runner.start(AgentRunRequest(input="input"))
    assert result.run.stop_reason is RunStopReason.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cancel", "deadline", "sdk", "unknown", "sdk_unknown"])
async def test_started_cleanup_retry_preserves_control_intent(tmp_path: Path, source: str) -> None:
    entered = asyncio.Event()
    service = ControlledService()
    service.release()
    cleanup = IrisCommandCleanupError("started cleanup pending")
    clock = FrozenClock()
    provider = StaticProvider(text_response())
    calls: list[str] = []

    async def before(
        event: HookEvent, *, cancellation: CancellationSignal | None
    ) -> HookInvocationOutcome:
        calls.append(event.event)
        entered.set()
        unknown = (
            IrisToolOutcomeUnknownError("started unknown", stop_receipt=service.receipt)
            if "unknown" in source
            else None
        )
        error: BaseException
        if source == "unknown":
            assert unknown is not None
            error = unknown
        else:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancelled:
                error = cancelled
        return HookInvocationOutcome(
            control=HookControl(
                "unknown" if source == "unknown" else "task_cancelled",
                error,
                stop_slot=CommandStopSlot(receipt=service.receipt, cleanup_error=cleanup),
                unknown_error=unknown,
            )
        )

    runner = _runner(
        tmp_path, CommandHookRegistration("run.started", "start", before), provider=provider
    )
    runner.clock = clock
    bind_service(runner, service)
    options = AgentRunOptions(limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=30)))
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="input", run_id="run"), options=options)
    )
    await entered.wait()
    if source == "deadline":
        clock.advance(seconds=31)
        await runner._command_deadline_due("run")
    elif source == "cancel":
        runner.request_cancel("run")
    elif source.startswith("sdk"):
        task.cancel()
    with pytest.raises(IrisCommandCleanupError, match="started cleanup pending"):
        await task
    pending = runner._command_lifecycle.pending["run"]
    expected = {
        "cancel": RunStopReason.CANCELLED,
        "deadline": RunStopReason.DEADLINE_EXCEEDED,
        "sdk": None,
        "unknown": RunStopReason.OUTCOME_UNKNOWN,
        "sdk_unknown": RunStopReason.OUTCOME_UNKNOWN,
    }[source]
    assert pending.stop_reason is expected
    assert pending.receipt is service.receipt
    assert not provider.requests
    result = await runner._command_lifecycle.join(pending)
    if expected is None:
        assert result is None
        result = await runner.recover(
            "run", expected_activation_id=runner.get_run("run").current_activation_id
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
    else:
        assert result is not None and result.run.stop_reason is expected
    assert calls == ["run.started"]
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_cancel_terminal_driver_ends_finished_without_second_settlement(
    tmp_path: Path, recover: bool
) -> None:
    class FailFinishOnceStore(InMemoryLifecycleStore):
        failed = False
        finishes = 0

        def finish_run(self, command: FinishRun) -> RunCommit:
            self.finishes += 1
            if recover and not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("finish crash")
            return super().finish_run(command)

    entered = asyncio.Event()
    cancelled = asyncio.Event()
    store = FailFinishOnceStore()

    async def finish(event: HookEvent) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    registration = HookRegistration(event="run.finished", name="finish", handler=finish)
    runner = _runner(tmp_path, registration, store=store)
    if recover:
        with pytest.raises(IrisRunPersistenceError):
            await runner.start(AgentRunRequest(input="input", run_id="run"))
        assert not entered.is_set()
        crashed = runner.get_run("run")
        runner = _runner(tmp_path, registration, store=store, provider=StaticProvider())
        task = asyncio.create_task(
            runner.recover("run", expected_activation_id=crashed.current_activation_id or "")
        )
    else:
        task = asyncio.create_task(runner.start(AgentRunRequest(input="input", run_id="run")))
    await entered.wait()
    durable = runner.get_result("run")
    assert durable is not None and durable.run.stop_reason is RunStopReason.COMPLETED
    with pytest.raises(IrisRunStateError):
        await runner.start(AgentRunRequest(input="next", run_id="next"))
    assert store.load_run("next") is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert runner.get_result("run") == durable
    assert not runner._command_lifecycle.pending
    assert store.finishes == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_finished_runs_before_slow_observer_delivery(tmp_path: Path) -> None:
    hook_entered = asyncio.Event()
    observer_entered = asyncio.Event()
    observer_release = asyncio.Event()

    async def finish(event: HookEvent) -> None:
        hook_entered.set()

    class Observer:
        async def on_event(self, event: RunEvent) -> None:
            observer_entered.set()
            await observer_release.wait()

    runner = _runner(
        tmp_path, HookRegistration(event="run.finished", name="finish", handler=finish)
    )
    runner.observers = [Observer()]
    runner._observer_locks = [asyncio.Lock()]
    task = asyncio.create_task(runner.start(AgentRunRequest(input="input")))
    await observer_entered.wait()
    assert hook_entered.is_set()
    observer_release.set()
    await task


@pytest.mark.asyncio
async def test_finished_cleanup_failure_reports_once_without_pending_retry(tmp_path: Path) -> None:
    async def unknown(event: HookEvent) -> None:
        raise IrisToolOutcomeUnknownError("finished effect unknown")

    runner = _runner(
        tmp_path, HookRegistration(event="run.finished", name="finish", handler=unknown)
    )
    facts: list[RunEvent | CommandCleanupFailed] = []
    runner._register_session_fact_callback("default", facts.append)
    service = ControlledService()
    service.fail = True
    bind_service(runner, service)
    with pytest.raises(IrisCommandCleanupError, match="stop failed"):
        await runner.start(AgentRunRequest(input="input", run_id="run"))
    assert runner.get_run("run").stop_reason is RunStopReason.COMPLETED
    assert not runner._command_lifecycle.pending
    failures = [fact for fact in facts if isinstance(fact, CommandCleanupFailed)]
    assert len(failures) == 1
    with pytest.raises(IrisCommandCleanupError, match="stop failed"):
        await runner.start(AgentRunRequest(input="next"))
    service.fail = False
    service.release()
    await runner.aclose()
