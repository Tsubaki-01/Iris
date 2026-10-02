"""有序 Hook 派发、只读输入与调用期控制的行为契约。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from iris.command.models import CommandStopReceipt, CommandStopSlot
from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisCommandCleanupError,
    IrisHookProtocolError,
    IrisToolOutcomeUnknownError,
)
from iris.hooks import (
    HookEvent,
    HookRegistration,
    RunFinishedEvent,
    RunStartedEvent,
    ToolAfterEvent,
    ToolAfterResult,
    ToolBeforeEvent,
    ToolBeforeResult,
)
from iris.hooks._dispatch_types import CommandHookRegistration, HookControl, HookInvocationOutcome
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import RunLimits, RunPhase, RunResult, RunSnapshot, RunStopReason, RunUsage
from iris.message import TextBlock
from iris.tools import ToolResult


def _before() -> ToolBeforeEvent:
    return ToolBeforeEvent(
        agent_id="agent",
        session_id="session",
        workspace="workspace",
        call_id="call",
        tool_name="exec_command",
        arguments={"nested": [1]},
    )


def _after() -> ToolAfterEvent:
    return ToolAfterEvent(
        agent_id="agent",
        session_id="session",
        workspace="workspace",
        call_id="call",
        tool_name="exec_command",
        arguments={"nested": [1]},
        result=ToolResult(
            tool_use_id="call", tool_name="exec_command", content=[TextBlock(text="body")]
        ),
        body_status="success",
    )


def _finished(reason: RunStopReason = RunStopReason.COMPLETED) -> RunFinishedEvent:
    now = datetime.now(UTC)
    run = RunSnapshot(
        run_id="run",
        session_id="session",
        agent_id="agent",
        phase=RunPhase.TERMINAL,
        stop_reason=reason,
        revision=1,
        limits=RunLimits(),
        usage=RunUsage(),
        checkpoint_sequence=0,
        last_event_sequence=1,
        created_at=now,
        started_at=now,
        updated_at=now,
        finished_at=now,
    )
    return RunFinishedEvent(
        agent_id="agent", session_id="session", workspace="workspace", result=RunResult(run=run)
    )


@pytest.mark.asyncio
async def test_ordered_exact_matching_and_independent_snapshots() -> None:
    seen: list[str] = []

    async def first(event: ToolAfterEvent) -> ToolAfterResult:
        seen.append("first")
        event.arguments["nested"].append(2)
        event.result.content[0].text = "changed"
        return ToolAfterResult(feedback="first feedback")

    async def last(event: ToolAfterEvent) -> ToolAfterResult:
        assert event.arguments == {"nested": [1]}
        assert event.result.model_content == "body"
        seen.append("last")
        return ToolAfterResult(feedback="last feedback")

    event = _after()
    dispatcher = HookDispatcher(
        [
            HookRegistration(event="tool.after", name="first", handler=first),
            HookRegistration(
                event="tool.after", name="config-alias", handler=last, tool_names=["exec.command"]
            ),
            HookRegistration(event="tool.before", name="wrong-event", handler=last),
            HookRegistration(
                event="tool.after", name="last", handler=last, tool_names=["exec_command"]
            ),
        ]
    )
    outcome = await dispatcher.dispatch(event)
    assert seen == ["first", "last"]
    assert outcome.feedback == ("first feedback", "last feedback")
    assert event.arguments == {"nested": [1]} and event.result.model_content == "body"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_before_rejection_or_failure_stops_remaining(failure: bool) -> None:
    seen: list[str] = []

    async def first(event: HookEvent) -> None:
        seen.append("first")

    async def reject(event: HookEvent) -> ToolBeforeResult:
        seen.append("reject")
        if failure:
            raise RuntimeError("invalid state")
        return ToolBeforeResult(deny_reason="not this call")

    outcome = await HookDispatcher(
        [
            HookRegistration(event="tool.before", name="first", handler=first),
            HookRegistration(event="tool.before", name="reject", handler=reject),
            HookRegistration(event="tool.before", name="last", handler=first),
        ]
    ).dispatch(_before())
    assert seen == ["first", "reject"]
    assert outcome.rejection.code == ("HOOK_ERROR" if failure else "HOOK_REJECTED")
    assert outcome.rejection.reason


@pytest.mark.asyncio
async def test_after_failure_keeps_ordered_feedback(caplog: pytest.LogCaptureFixture) -> None:
    async def feedback(event: HookEvent) -> ToolAfterResult:
        return ToolAfterResult(feedback="feedback")

    async def fail(event: HookEvent) -> None:
        raise RuntimeError("ordinary failure")

    outcome = await HookDispatcher(
        [
            HookRegistration(event="tool.after", name=str(i), handler=handler)
            for i, handler in enumerate([feedback, fail, feedback])
        ]
    ).dispatch(_after())
    assert outcome.feedback == ("feedback", "feedback")
    assert outcome.control is None and "ordinary failure" in caplog.text


@pytest.mark.asyncio
async def test_partial_feedback_is_returned_with_original_control() -> None:
    error = IrisCancellationRequestedError("cancel")

    async def feedback(event: HookEvent) -> ToolAfterResult:
        return ToolAfterResult(feedback="kept")

    async def cancel(event: HookEvent) -> None:
        raise error

    outcome = await HookDispatcher(
        [
            HookRegistration(event="tool.after", name=str(i), handler=handler)
            for i, handler in enumerate([feedback, feedback, cancel, feedback])
        ]
    ).dispatch(_after())
    assert outcome.feedback == ("kept", "kept")
    assert outcome.control.error is error
    assert outcome.control.origin == "run_cancelled"


class _Signal:
    requested = False

    def raise_if_requested(self) -> None:
        if self.requested:
            raise IrisCancellationRequestedError("requested")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["signal", "task"])
async def test_external_control_cancels_and_drains_once(source: str) -> None:
    entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    signal = _Signal()
    cleanup_cancelled = False

    async def handler(event: HookEvent) -> ToolAfterResult:
        nonlocal cleanup_cancelled
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cleanup_cancelled = True
                raise
            return ToolAfterResult(feedback="too late")

    dispatcher = HookDispatcher(
        [HookRegistration(event="tool.after", name="wait", handler=handler)]
    )
    task = asyncio.create_task(dispatcher.dispatch(_after(), cancellation=signal))
    await entered.wait()
    if source == "signal":
        signal.requested = True
    else:
        task.cancel()
    try:
        await asyncio.wait_for(cleanup.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not cleanup_cancelled
    finally:
        release.set()
    outcome = await task
    assert outcome.feedback == ()
    assert outcome.control.origin == ("run_cancelled" if source == "signal" else "task_cancelled")


@pytest.mark.asyncio
@pytest.mark.parametrize("swallow", [False, True])
async def test_python_timeout_discards_late_feedback(swallow: bool) -> None:
    async def handler(event: HookEvent) -> ToolAfterResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not swallow:
                raise
            return ToolAfterResult(feedback="too late")

    outcome = await HookDispatcher(
        [
            HookRegistration(
                event="tool.after", name="timeout", handler=handler, timeout_seconds=0.01
            )
        ]
    ).dispatch(_after())
    assert outcome.feedback == () and outcome.control is None


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_run_non_none_python_output_is_protocol_error(
    caplog: pytest.LogCaptureFixture, started: bool
) -> None:
    async def bad(event: HookEvent) -> dict[str, object]:
        return {}

    event = _finished()
    if started:
        run = event.result.run.model_copy(
            update={
                "phase": RunPhase.ACTIVE,
                "stop_reason": None,
                "finished_at": None,
                "current_activation_id": "activation",
            }
        )
        event = RunStartedEvent(
            agent_id="agent", session_id="session", workspace="workspace", run=run, input="hello"
        )
    await HookDispatcher([HookRegistration(event=event.event, name="bad", handler=bad)]).dispatch(
        event
    )
    assert IrisHookProtocolError.runtime_error_code in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [RunStopReason.COMPLETED, RunStopReason.CANCELLED])
async def test_finished_command_registration_has_explicit_kind(reason: RunStopReason) -> None:
    seen: list[str] = []

    async def python(event: HookEvent) -> None:
        seen.append("python")

    async def command(event: HookEvent, *, cancellation: object = None) -> HookInvocationOutcome:
        seen.append("command")
        return HookInvocationOutcome()

    dispatcher = HookDispatcher(
        [
            HookRegistration(event="run.finished", name="python", handler=python),
            CommandHookRegistration(event="run.finished", name="command", handler=command),
        ]
    )
    await dispatcher.dispatch(_finished(reason))
    assert seen == (["python", "command"] if reason is RunStopReason.COMPLETED else ["python"])


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["signal", "task"])
@pytest.mark.parametrize("kind", ["unknown", "cleanup"])
async def test_control_preserves_error_reported_while_draining(source: str, kind: str) -> None:
    entered = asyncio.Event()
    signal = _Signal()
    receipt = CommandStopReceipt("service", "stop")
    error = (
        IrisToolOutcomeUnknownError("unknown", stop_receipt=receipt)
        if kind == "unknown"
        else IrisCommandCleanupError("cleanup")
    )

    async def handler(event: HookEvent) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise error from None

    task = asyncio.create_task(
        HookDispatcher(
            [HookRegistration(event="tool.after", name="drain", handler=handler)]
        ).dispatch(_after(), cancellation=signal)
    )
    await entered.wait()
    if source == "signal":
        signal.requested = True
    else:
        task.cancel()
    outcome = await task
    assert outcome.control.origin == ("run_cancelled" if source == "signal" else "task_cancelled")
    if kind == "unknown":
        assert outcome.control.unknown_error is error
        assert outcome.control.stop_slot.receipt is receipt
    else:
        assert outcome.control.stop_slot.cleanup_error is error
    assert outcome.feedback == ()


@pytest.mark.asyncio
async def test_command_control_return_keeps_locked_cancel_and_original_slot() -> None:
    entered = asyncio.Event()
    error = IrisToolOutcomeUnknownError("unknown")
    slot = CommandStopSlot(receipt=CommandStopReceipt("service", "stop"))

    async def command(event: HookEvent, *, cancellation: object = None) -> HookInvocationOutcome:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return HookInvocationOutcome(
                control=HookControl(
                    "unknown", error, stop_slot=slot, call_id="hook-call", unknown_error=error
                )
            )

    task = asyncio.create_task(
        HookDispatcher(
            [CommandHookRegistration(event="tool.after", name="command", handler=command)]
        ).dispatch(_after())
    )
    await entered.wait()
    task.cancel()
    outcome = await task
    assert outcome.control.origin == "task_cancelled"
    assert isinstance(outcome.control.error, asyncio.CancelledError)
    assert outcome.control.unknown_error is error
    assert outcome.control.stop_slot is slot
    assert outcome.control.call_id == "hook-call"


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["timeout", "external"])
async def test_python_budget_and_external_cancel_do_not_recancel_cleanup(first: str) -> None:
    entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    interrupted = False

    async def handler(event: HookEvent) -> ToolAfterResult:
        nonlocal interrupted
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted = True
                raise
            return ToolAfterResult(feedback="late")

    task = asyncio.create_task(
        HookDispatcher(
            [
                HookRegistration(
                    event="tool.after", name="budget", handler=handler, timeout_seconds=0.02
                )
            ]
        ).dispatch(_after())
    )
    await entered.wait()
    if first == "external":
        task.cancel()
    await asyncio.wait_for(cleanup.wait(), 1)
    if first == "timeout":
        task.cancel()
    try:
        await asyncio.sleep(0.04)
        task.cancel()
        await asyncio.sleep(0)
        assert not interrupted
    finally:
        release.set()
    outcome = await task
    assert outcome.feedback == () and outcome.control.origin == "task_cancelled"


@pytest.mark.asyncio
async def test_requested_signal_skips_handler_and_beats_simultaneous_failure() -> None:
    signal = _Signal()
    signal.requested = True
    entered = False

    async def handler(event: HookEvent) -> None:
        nonlocal entered
        entered = True
        signal.requested = True
        raise RuntimeError("ordinary")

    dispatcher = HookDispatcher(
        [HookRegistration(event="tool.before", name="before", handler=handler)]
    )
    outcome = await dispatcher.dispatch(_before(), cancellation=signal)
    assert not entered and outcome.control.origin == "run_cancelled"
    signal.requested = False
    outcome = await dispatcher.dispatch(_before(), cancellation=signal)
    assert entered and outcome.control.origin == "run_cancelled"
    assert outcome.rejection is None


@pytest.mark.asyncio
async def test_before_timeout_is_hook_error_without_control() -> None:
    async def handler(event: HookEvent) -> None:
        await asyncio.Event().wait()

    outcome = await HookDispatcher(
        [
            HookRegistration(
                event="tool.before", name="timeout", handler=handler, timeout_seconds=0.01
            )
        ]
    ).dispatch(_before())
    assert outcome.rejection.code == "HOOK_ERROR" and outcome.control is None


@pytest.mark.asyncio
async def test_wrong_tool_event_result_is_protocol_error(caplog: pytest.LogCaptureFixture) -> None:
    async def handler(event: HookEvent) -> ToolBeforeResult:
        return ToolBeforeResult(deny_reason="invalid after")

    outcome = await HookDispatcher(
        [HookRegistration(event="tool.after", name="wrong", handler=handler)]
    ).dispatch(_after())
    assert outcome.feedback == () and outcome.rejection is None
    assert "HOOK_PROTOCOL_ERROR" in caplog.text


@pytest.mark.asyncio
async def test_dispatches_do_not_share_a_global_serial_lock() -> None:
    entered = 0
    both = asyncio.Event()

    async def handler(event: HookEvent) -> ToolAfterResult:
        nonlocal entered
        entered += 1
        if entered == 2:
            both.set()
        await both.wait()
        return ToolAfterResult(feedback=str(entered))

    dispatcher = HookDispatcher(
        [HookRegistration(event="tool.after", name="shared", handler=handler)]
    )
    results = await asyncio.wait_for(
        asyncio.gather(dispatcher.dispatch(_after()), dispatcher.dispatch(_after())), 1
    )
    assert entered == 2 and all(result.feedback == ("2",) for result in results)
