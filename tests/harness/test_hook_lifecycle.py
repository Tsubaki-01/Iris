"""root 共享 finished owner 的登记、完成通知与后终态资源交接。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.command import (
    CommandBinding,
    CommandConfig,
    CommandEnvironment,
    CommandMode,
    CommandScope,
    CommandStopReceipt,
)
from iris.exceptions import IrisCommandCleanupError, IrisRunStateError, IrisToolOutcomeUnknownError
from iris.harness import AgentRunner
from iris.harness._hooks import HookLifecycle
from iris.harness.streaming import CommandCleanupFailed
from iris.hooks import HookEvent, HookRegistration
from iris.hooks._dispatch_types import CommandHookRegistration, HookInvocationOutcome
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import (
    AgentRunRequest,
    RunErrorInfo,
    RunLimits,
    RunPhase,
    RunResult,
    RunSnapshot,
    RunStopReason,
    RunUsage,
)
from iris.store import InMemoryLifecycleStore

from .fakes import build_runtime


def _result(reason: RunStopReason = RunStopReason.COMPLETED) -> RunResult:
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
    return RunResult(
        run=run,
        error=RunErrorInfo(code="TEST_FAILURE", message="test", source="runtime")
        if reason in {RunStopReason.FAILED, RunStopReason.OUTCOME_UNKNOWN}
        else None,
    )


def _runner(tmp_path: Path, dispatcher: HookDispatcher) -> AgentRunner:
    runtime = build_runtime(tmp_path)
    runtime.environment.hook_dispatcher = dispatcher
    return AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())


@pytest.mark.asyncio
async def test_registration_blocks_admission_before_publication_and_waiter_is_independent(
    tmp_path: Path,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(event: HookEvent) -> None:
        entered.set()
        await release.wait()

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="wait", handler=handler)]),
    )
    lifecycle = HookLifecycle(runner)
    result = _result()
    notified: list[bool] = []
    lifecycle.subscribe("session", lambda: notified.append(True))
    entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    assert entry is not None and lifecycle.has_pending("session")
    assert not entered.is_set()
    with pytest.raises(IrisRunStateError):
        lifecycle.check_admission("session")
    lifecycle.check_admission("other")
    lifecycle.publish_finished(entry, result)
    await entered.wait()
    waiter = asyncio.create_task(lifecycle.wait_session("session"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert lifecycle.has_pending("session")
    release.set()
    await lifecycle.join_finished("run")
    assert not lifecycle.has_pending("session") and notified == [True]
    lifecycle.check_admission("session")


@pytest.mark.asyncio
async def test_no_applicable_finished_does_not_register_owner(tmp_path: Path) -> None:
    async def command(event: HookEvent, *, cancellation: object = None) -> HookInvocationOutcome:
        pytest.fail("非completed不运行命令")

    runner = _runner(
        tmp_path,
        HookDispatcher(
            [CommandHookRegistration(event="run.finished", name="command", handler=command)]
        ),
    )
    lifecycle = HookLifecycle(runner)
    result = _result(RunStopReason.CANCELLED)
    assert (
        lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason) is None
    )
    assert not lifecycle.has_pending("session")
    await lifecycle.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", list(RunStopReason))
async def test_python_finished_covers_all_reasons_command_only_completed(
    tmp_path: Path, reason: RunStopReason
) -> None:
    seen: list[tuple[str, RunStopReason]] = []

    async def python(event: HookEvent) -> None:
        seen.append(("python", event.result.run.stop_reason))

    async def command(event: HookEvent, *, cancellation: object = None) -> HookInvocationOutcome:
        seen.append(("command", event.result.run.stop_reason))
        return HookInvocationOutcome()

    runner = _runner(
        tmp_path,
        HookDispatcher(
            [
                HookRegistration(event="run.finished", name="python", handler=python),
                CommandHookRegistration(event="run.finished", name="command", handler=command),
            ]
        ),
    )
    lifecycle = HookLifecycle(runner)
    result = _result(reason)
    entry = lifecycle.register_finished(runner, result.run, stop_reason=reason)
    lifecycle.publish_finished(entry, result)
    await lifecycle.join_finished("run")
    expected = [("python", reason)]
    if reason is RunStopReason.COMPLETED:
        expected.append(("command", reason))
    assert seen == expected
    assert lifecycle._finished == {}


@pytest.mark.asyncio
async def test_cancel_driver_stops_owner_but_does_not_cancel_other_waiters(tmp_path: Path) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def handler(event: HookEvent) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="wait", handler=handler)]),
    )
    lifecycle = HookLifecycle(runner)
    result = _result()
    entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    lifecycle.publish_finished(entry, result)
    driver = asyncio.create_task(lifecycle.join_finished("run"))
    await entered.wait()
    observer = asyncio.create_task(lifecycle.wait_session("session"))
    driver.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driver
    await observer
    assert cleaned.is_set() and lifecycle.admission_error is None


class _Service:
    def __init__(self) -> None:
        self.receipt = CommandStopReceipt("service", "stop")
        self.error: IrisCommandCleanupError | None = None
        self.stops: list[CommandScope] = []
        self.receipts: list[CommandStopReceipt] = []

    def stop(self, scope: CommandScope) -> _Service:
        self.stops.append(scope)
        return self

    async def wait_drained(self, receipt: CommandStopReceipt | None = None) -> CommandStopReceipt:
        if receipt is not None:
            self.receipts.append(receipt)
        if self.error is not None:
            raise self.error
        return self.receipt


def _bind(runner: AgentRunner, service: _Service) -> None:
    runner.runtime.environment.command_binding = CommandBinding(
        config=CommandConfig(),
        service=service,
        environment=CommandEnvironment("test", CommandMode.NATIVE, "test", "shell"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [False, True])
async def test_finished_unknown_drains_without_changing_result(
    tmp_path: Path, receipt: bool
) -> None:
    service = _Service()

    async def handler(event: HookEvent) -> None:
        raise IrisToolOutcomeUnknownError(
            "unknown", stop_receipt=service.receipt if receipt else None
        )

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="unknown", handler=handler)]),
    )
    _bind(runner, service)
    lifecycle = HookLifecycle(runner)
    result = _result()
    entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    lifecycle.publish_finished(entry, result)
    await lifecycle.join_finished("run")
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert lifecycle.admission_error is None
    assert service.receipts == ([service.receipt] if receipt else [])
    assert service.stops == ([] if receipt else [CommandScope("run", "session")])


@pytest.mark.asyncio
async def test_root_failure_wakes_other_session_and_close_stays_available(tmp_path: Path) -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()
    service = _Service()
    service.error = IrisCommandCleanupError("not drained")

    async def handler(event: HookEvent) -> None:
        if event.session_id == "other":
            blocked.set()
            await release.wait()
        else:
            raise IrisToolOutcomeUnknownError("unknown", stop_receipt=service.receipt)

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="hook", handler=handler)]),
    )
    _bind(runner, service)
    lifecycle = HookLifecycle(runner)
    other = _result().model_copy(
        update={"run": _result().run.model_copy(update={"run_id": "other", "session_id": "other"})}
    )
    other_entry = lifecycle.register_finished(
        runner, other.run, stop_reason=RunStopReason.COMPLETED
    )
    lifecycle.publish_finished(other_entry, other)
    await blocked.wait()
    waiter = asyncio.create_task(lifecycle.wait_session("other"))
    result = _result()
    entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    lifecycle.publish_finished(entry, result)
    with pytest.raises(IrisCommandCleanupError) as caught:
        await lifecycle.join_finished("run")
    assert caught.value is service.error is lifecycle.admission_error
    with pytest.raises(IrisCommandCleanupError):
        await asyncio.wait_for(waiter, 1)
    with pytest.raises(IrisCommandCleanupError) as admission:
        lifecycle.check_admission("new-session")
    assert admission.value is service.error
    close = asyncio.create_task(lifecycle.aclose())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await close


@pytest.mark.asyncio
async def test_cleanup_error_has_priority_over_driver_cancellation(tmp_path: Path) -> None:
    entered = asyncio.Event()
    service = _Service()
    service.error = IrisCommandCleanupError("not drained")

    async def handler(event: HookEvent) -> None:
        entered.set()
        await asyncio.Event().wait()

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="wait", handler=handler)]),
    )
    _bind(runner, service)
    lifecycle = HookLifecycle(runner)
    result = _result()
    entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    lifecycle.publish_finished(entry, result)
    driver = asyncio.create_task(lifecycle.join_finished("run"))
    await entered.wait()
    driver.cancel()
    with pytest.raises(IrisCommandCleanupError) as caught:
        await driver
    assert caught.value is service.error
    assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert entry.completion.is_set()


@pytest.mark.asyncio
async def test_withdraw_only_own_registration_and_closing_scope_is_temporary(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def handler(event: HookEvent) -> None:
        calls.append(event.run_id)

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="hook", handler=handler)]),
    )
    lifecycle = HookLifecycle(runner)
    result = _result()
    entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    duplicate = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
    assert duplicate is None
    lifecycle.withdraw_finished(duplicate)
    assert lifecycle.has_pending("session")
    lifecycle.withdraw_finished(entry)
    await entry.task
    assert not calls and not lifecycle.has_pending("session")
    with lifecycle.cancelling_session("session"):
        entry = lifecycle.register_finished(runner, result.run, stop_reason=result.run.stop_reason)
        await asyncio.sleep(0)
        assert lifecycle.has_pending("session")
        lifecycle.publish_finished(entry, result)
        await lifecycle.join_finished("run")
    assert not calls
    next_result = result.model_copy(
        update={"run": result.run.model_copy(update={"run_id": "next"})}
    )
    entry = lifecycle.register_finished(
        runner, next_result.run, stop_reason=RunStopReason.COMPLETED
    )
    lifecycle.publish_finished(entry, next_result)
    await lifecycle.join_finished("next")
    assert calls == ["next"]


@pytest.mark.asyncio
async def test_completed_owners_release_results_and_exact_registrations(tmp_path: Path) -> None:
    calls: list[str] = []

    async def handler(event: HookEvent) -> None:
        calls.append(event.run_id)

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="finished", handler=handler)]),
    )
    lifecycle = HookLifecycle(runner)
    result = _result()
    abandoned = lifecycle.register_finished(runner, result.run, stop_reason=RunStopReason.COMPLETED)
    lifecycle.withdraw_finished(abandoned)
    replacement = lifecycle.register_finished(
        runner, result.run, stop_reason=RunStopReason.COMPLETED
    )
    await abandoned.task
    assert lifecycle._finished["run"] is replacement
    lifecycle.publish_finished(replacement, result)
    await lifecycle.join_finished("run")
    assert lifecycle._finished == {}
    for index in range(5):
        next_result = result.model_copy(
            update={"run": result.run.model_copy(update={"run_id": str(index)})}
        )
        entry = lifecycle.register_finished(
            runner, next_result.run, stop_reason=RunStopReason.COMPLETED
        )
        lifecycle.publish_finished(entry, next_result)
        await lifecycle.join_finished(str(index))
        assert lifecycle._finished == {}
    assert calls == ["run", "0", "1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_cleanup_notification_cancel_of_sdk_driver_keeps_resource_error(
    tmp_path: Path,
) -> None:
    service = _Service()
    service.error = IrisCommandCleanupError("actual drain failed")

    async def handler(event: HookEvent) -> None:
        raise IrisCommandCleanupError("hook cleanup failed")

    runner = _runner(
        tmp_path,
        HookDispatcher([HookRegistration(event="run.finished", name="cleanup", handler=handler)]),
    )
    _bind(runner, service)
    driver: asyncio.Task[RunResult] | None = None

    def on_fact(fact: object) -> None:
        if isinstance(fact, CommandCleanupFailed):
            assert driver is not None
            driver.cancel()

    runner._register_session_fact_callback("session", on_fact)
    driver = asyncio.create_task(
        runner.start(AgentRunRequest(input="hello", session_id="session", run_id="run"))
    )
    with pytest.raises(IrisCommandCleanupError) as caught:
        await driver
    assert caught.value is service.error
    assert runner.get_result("run").run.stop_reason is RunStopReason.COMPLETED
    assert runner._hook_lifecycle.admission_error is service.error
