"""命令 Hook 的唯一输出边界与停止事实交接。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.command.config import CommandConfig
from iris.command.models import (
    CommandEnvironment,
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
)
from iris.command.service import CommandBinding
from iris.exceptions import (
    IrisCommandCleanupError,
    IrisHookError,
    IrisHookProtocolError,
    IrisToolOutcomeUnknownError,
)
from iris.hooks import (
    HookEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolAfterEvent,
    ToolAfterResult,
    ToolBeforeEvent,
    ToolBeforeResult,
)
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import RunLimits, RunPhase, RunResult, RunSnapshot, RunStopReason, RunUsage
from iris.message import TextBlock
from iris.tools import ToolResult


def _event(kind: str = "tool.after") -> HookEvent:
    common = dict(
        agent_id="agent",
        session_id="session",
        run_id="run",
        activation_id="activation",
        workspace="workspace",
    )
    if kind in {"tool.before", "tool.after"}:
        values = common | dict(
            call_id="tool-call", tool_name="exec_command", arguments={"command": "中文命令"}
        )
        if kind == "tool.before":
            return ToolBeforeEvent(**values)
        return ToolAfterEvent(
            **values,
            result=ToolResult(
                tool_use_id="tool-call", tool_name="exec_command", content=[TextBlock(text="body")]
            ),
            body_status="success",
        )
    now = datetime.now(UTC)
    terminal = kind == "run.finished"
    run = RunSnapshot(
        run_id="run",
        session_id="session",
        agent_id="agent",
        phase=RunPhase.TERMINAL if terminal else RunPhase.ACTIVE,
        stop_reason=RunStopReason.COMPLETED if terminal else None,
        current_activation_id=None if terminal else "activation",
        revision=1,
        limits=RunLimits(),
        usage=RunUsage(),
        checkpoint_sequence=0,
        last_event_sequence=1,
        created_at=now,
        started_at=now,
        updated_at=now,
        finished_at=now if terminal else None,
    )
    if terminal:
        return RunFinishedEvent(**common, result=RunResult(run=run))
    return RunStartedEvent(**common, run=run, input="中文输入")


def _outcome(
    stdout: str = '{"feedback":"ok"}',
    *,
    status: CommandStatus = CommandStatus.EXITED,
    receipt: CommandStopReceipt | None = None,
    exit_code: int = 0,
    stderr: str = "",
) -> CommandOutcome:
    return CommandOutcome(
        mode=CommandMode.NATIVE,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        output_stats=CommandOutputStats(
            len(stdout.encode()),
            len(stderr.encode()),
            len(stdout.encode()),
            len(stderr.encode()),
            frozenset(),
        ),
        duration_seconds=0.01,
        cwd="workspace",
        stop_receipt=receipt,
    )


class _Service:
    def __init__(self, outcome: CommandOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[CommandScope, CommandRequest]] = []
        self.drained: list[CommandStopReceipt] = []
        self.drain_error: BaseException | None = None

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        self.calls.append((scope, request))
        await asyncio.sleep(0)
        return self.outcome

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        self.drained.append(receipt)
        if self.drain_error is not None:
            raise self.drain_error


def _adapter(service: _Service, workspace: Path, timeout: float = 10) -> CommandHookAdapter:
    binding = CommandBinding(
        config=CommandConfig(timeout_seconds=0.01),
        service=service,
        environment=CommandEnvironment("test", CommandMode.NATIVE, "test", "shell"),
    )
    return CommandHookAdapter(
        binding=binding, workspace=workspace, command="run-hook", timeout_seconds=timeout
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,stdout,expected",
    [
        ("tool.before", '{"deny_reason":"不执行"}', ToolBeforeResult(deny_reason="不执行")),
        ("tool.after", '{"feedback":"中文反馈"}', ToolAfterResult(feedback="中文反馈")),
        ("run.started", "{}", None),
        ("run.finished", "{}", None),
        ("tool.before", "{}", None),
        ("tool.after", "{}", None),
    ],
)
async def test_valid_event_output_and_utf8_stdin(
    tmp_path: Path, kind: str, stdout: str, expected: object
) -> None:
    service = _Service(_outcome(stdout))
    result = await _adapter(service, tmp_path, 30)(_event(kind), cancellation=None)
    scope, request = service.calls[0]
    assert result.result == expected and result.control is None
    assert scope == CommandScope("run", "session")
    assert request.call_id.startswith("hook_") and request.call_id != "tool-call"
    assert request.cwd == tmp_path and request.timeout_seconds == 30
    assert request.payload.command == "run-hook"
    payload = json.loads(request.stdin.decode("utf-8"))
    assert payload["event"] == kind and payload["run_id"] == "run"
    assert b"\\u4e2d" not in request.stdin


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "null",
        "[]",
        '"text"',
        "{}{}",
        '{"feedback":""}',
        '{"feedback":"ok","extra":1}',
        '{"deny_reason":"wrong-event"}',
    ],
)
async def test_invalid_stdout_is_protocol_error(tmp_path: Path, stdout: str) -> None:
    service = _Service(_outcome(stdout))
    with pytest.raises(IrisHookProtocolError):
        await _adapter(service, tmp_path)(_event(), cancellation=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run.started", "run.finished"])
async def test_run_event_accepts_only_empty_object(tmp_path: Path, kind: str) -> None:
    with pytest.raises(IrisHookProtocolError):
        await _adapter(_Service(_outcome()), tmp_path)(_event(kind), cancellation=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["stdout", "stderr", "drain_timeout", "stream_error", "stream_closed"]
)
async def test_output_completeness_distinguishes_stdout_and_stderr(
    tmp_path: Path, reason: str
) -> None:
    outcome = _outcome(stderr="diagnostic")
    stats = outcome.output_stats
    stats = replace(
        stats,
        truncation_reasons=frozenset({"byte_limit" if reason in {"stdout", "stderr"} else reason}),
    )
    if reason == "stdout":
        stats = replace(stats, stdout_bytes=stats.stdout_bytes + 1)
    elif reason == "stderr":
        stats = replace(stats, stderr_bytes=stats.stderr_bytes + 1)
    adapter = _adapter(_Service(replace(outcome, output_stats=stats)), tmp_path)
    if reason == "stderr":
        assert (await adapter(_event(), cancellation=None)).result == ToolAfterResult(feedback="ok")
    else:
        with pytest.raises(IrisHookProtocolError):
            await adapter(_event(), cancellation=None)


@pytest.mark.asyncio
async def test_stderr_is_logged_not_feedback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    result = await _adapter(_Service(_outcome(stderr="diagnostic")), tmp_path)(
        _event(), cancellation=None
    )
    assert result.result.feedback == "ok" and "diagnostic" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,exit_code", [(CommandStatus.EXITED, 2), (CommandStatus.TIMED_OUT, 0)]
)
async def test_ordinary_failure_drains_receipt_before_raising(
    tmp_path: Path, status: CommandStatus, exit_code: int
) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = _Service(_outcome(status=status, receipt=receipt, exit_code=exit_code))
    with pytest.raises(IrisHookError):
        await _adapter(service, tmp_path)(_event(), cancellation=None)
    assert service.drained == [receipt]


@pytest.mark.asyncio
async def test_success_with_receipt_remains_environment_control(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = _Service(_outcome(receipt=receipt))
    outcome = await _adapter(service, tmp_path)(_event(), cancellation=None)
    assert outcome.result is None and outcome.control.origin == "environment_interrupted"
    assert outcome.control.stop_slot.receipt is receipt


@pytest.mark.asyncio
async def test_undrained_receipt_stops_following_command(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = _Service(_outcome(status=CommandStatus.TIMED_OUT, receipt=receipt))
    error = IrisCommandCleanupError("not drained")
    service.drain_error = error
    adapter = _adapter(service, tmp_path)
    outcome = await HookDispatcher(
        [
            CommandHookRegistration(event="tool.after", name=str(i), handler=adapter)
            for i in range(2)
        ]
    ).dispatch(_event())
    assert len(service.calls) == 1 and outcome.feedback == ()
    assert outcome.control.stop_slot.receipt is receipt
    assert outcome.control.stop_slot.cleanup_error is error


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CommandStatus.CANCELLED, CommandStatus.EXITED])
@pytest.mark.parametrize("has_receipt", [False, True])
async def test_service_consumed_cancel_cannot_turn_into_success(
    tmp_path: Path, status: CommandStatus, has_receipt: bool
) -> None:
    entered = asyncio.Event()

    class CancelService(_Service):
        async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
            self.calls.append((scope, request))
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return self.outcome

    receipt = CommandStopReceipt("service", "stop") if has_receipt else None
    service = CancelService(_outcome(status=status, receipt=receipt))
    task = asyncio.create_task(
        HookDispatcher(
            [
                CommandHookRegistration(
                    event="tool.after", name="cancel", handler=_adapter(service, tmp_path)
                )
            ]
        ).dispatch(_event())
    )
    await entered.wait()
    task.cancel()
    outcome = await task
    assert outcome.feedback == () and outcome.control.origin == "task_cancelled"
    assert outcome.control.stop_slot.receipt is receipt
    assert outcome.control.stop_slot.status is status


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cleanup"])
async def test_execute_errors_preserve_unique_stop_facts(tmp_path: Path, kind: str) -> None:
    receipt = CommandStopReceipt("service", "stop")
    command_outcome = _outcome(status=CommandStatus.CANCELLED, receipt=receipt)
    error = (
        IrisToolOutcomeUnknownError("unknown", stop_receipt=receipt)
        if kind == "unknown"
        else IrisCommandCleanupError("cleanup", command_outcome=command_outcome)
    )

    class ErrorService(_Service):
        async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
            self.calls.append((scope, request))
            raise error

    outcome = await _adapter(ErrorService(command_outcome), tmp_path)(_event(), cancellation=None)
    assert outcome.result is None and outcome.control.error is error
    assert outcome.control.stop_slot.receipt is receipt
    if kind == "unknown":
        assert outcome.control.unknown_error is error
    else:
        assert outcome.control.stop_slot.cleanup_error is error
        assert outcome.control.stop_slot.status is CommandStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancellation_while_draining_preserves_pending_receipt(tmp_path: Path) -> None:
    entered = asyncio.Event()
    receipt = CommandStopReceipt("service", "stop")

    class DrainService(_Service):
        async def wait_drained(self, receipt: CommandStopReceipt) -> None:
            self.drained.append(receipt)
            entered.set()
            await asyncio.Event().wait()

    service = DrainService(_outcome(status=CommandStatus.TIMED_OUT, receipt=receipt))
    task = asyncio.create_task(
        HookDispatcher(
            [
                CommandHookRegistration(
                    event="tool.after", name="drain", handler=_adapter(service, tmp_path)
                )
            ]
        ).dispatch(_event())
    )
    await entered.wait()
    task.cancel()
    outcome = await task
    assert outcome.feedback == () and outcome.control.origin == "task_cancelled"
    assert outcome.control.stop_slot.receipt is receipt


@pytest.mark.asyncio
async def test_environment_interruption_is_control_even_with_valid_json(tmp_path: Path) -> None:
    service = _Service(_outcome(status=CommandStatus.ENVIRONMENT_INTERRUPTED))
    outcome = await _adapter(service, tmp_path)(_event(), cancellation=None)
    assert outcome.result is None and outcome.control.origin == "environment_interrupted"


@pytest.mark.asyncio
async def test_concurrent_calls_have_separate_hook_ids(tmp_path: Path) -> None:
    service = _Service(_outcome())
    adapter = _adapter(service, tmp_path)
    results = await asyncio.gather(*[adapter(_event(), cancellation=None) for _ in range(3)])
    assert len({request.call_id for _, request in service.calls}) == 3
    assert all(result.result.feedback == "ok" for result in results)
