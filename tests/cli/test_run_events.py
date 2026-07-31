from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import iris.cli.run as run_module
from iris.cli.run import RunEventsOptions, run_events
from iris.cli.run_output import RunCommandRenderer
from iris.exceptions import IrisRunNotFoundError
from iris.lifecycle import RunEvent, RunEventKind


class _RecordingEventsRunner:
    def __init__(
        self,
        *,
        events: list[RunEvent] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.events = events or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def list_events(self, run_id: str, *, after_sequence: int = 0) -> list[RunEvent]:
        self.calls.append((run_id, after_sequence))
        if self.error is not None:
            raise self.error
        return list(self.events)


def _event(
    sequence: int,
    *,
    kind: RunEventKind = RunEventKind.ACTIVATION_STARTED,
    activation_id: str | None = None,
    step_index: int | None = None,
    correlation_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> RunEvent:
    return RunEvent(
        run_id="run-1",
        session_id="session-1",
        sequence=sequence,
        kind=kind,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        activation_id=activation_id,
        step_index=step_index,
        correlation_id=correlation_id,
        payload=payload or {},
    )


def _renderer(*, json_output: bool) -> tuple[RunCommandRenderer, io.StringIO, io.StringIO]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    return (
        RunCommandRenderer(json_output=json_output, stdout=stdout, stderr=stderr),
        stdout,
        stderr,
    )


def _patch_runner(monkeypatch: pytest.MonkeyPatch, runner: _RecordingEventsRunner) -> None:
    monkeypatch.setattr(run_module, "_build_non_executing_runner", lambda options: runner)


def test_run_events_options_validate_identity_and_cursor() -> None:
    with pytest.raises(ValueError, match="run_id 不能为空"):
        RunEventsOptions(config_path=Path("agent.yaml"), run_id="  ")

    with pytest.raises(ValueError, match="after_sequence 必须大于等于 0"):
        RunEventsOptions(
            config_path=Path("agent.yaml"),
            run_id="run-1",
            after_sequence=-1,
        )


def test_run_events_calls_public_runner_once_and_emits_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _RecordingEventsRunner(
        events=[
            _event(8, kind=RunEventKind.ACTIVATION_STARTED),
            _event(9, kind=RunEventKind.MODEL_STEP_RESERVED),
        ]
    )
    _patch_runner(monkeypatch, runner)
    renderer, stdout, stderr = _renderer(json_output=True)

    code = run_events(
        RunEventsOptions(
            config_path=Path("agent.yaml"),
            run_id=" run-1 ",
            after_sequence=7,
            json_output=True,
        ),
        renderer=renderer,
    )

    assert code == 0
    assert runner.calls == [("run-1", 7)]
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert list(payload) == [
        "ok",
        "command",
        "run_id",
        "after_sequence",
        "next_after_sequence",
        "events",
        "error",
    ]
    assert payload["ok"] is True
    assert payload["command"] == "events"
    assert payload["run_id"] == "run-1"
    assert payload["after_sequence"] == 7
    assert payload["next_after_sequence"] == 9
    assert [event["sequence"] for event in payload["events"]] == [8, 9]
    assert payload["error"] is None


def test_run_events_unknown_run_emits_normalized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _RecordingEventsRunner(
        error=IrisRunNotFoundError("run 不存在", run_id="missing")
    )
    _patch_runner(monkeypatch, runner)
    renderer, stdout, stderr = _renderer(json_output=True)

    code = run_events(
        RunEventsOptions(
            config_path=Path("agent.yaml"),
            run_id="missing",
            after_sequence=4,
            json_output=True,
        ),
        renderer=renderer,
    )

    assert code == 1
    assert runner.calls == [("missing", 4)]
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload["ok"] is False
    assert payload["events"] == []
    assert payload["next_after_sequence"] == 4
    assert payload["error"] == {
        "code": "RUN_NOT_FOUND",
        "source": "lifecycle",
        "message": "run 不存在",
        "details": {"run_id": "missing"},
    }


def test_run_events_human_renderer_preserves_event_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _RecordingEventsRunner(
        events=[
            _event(
                3,
                kind=RunEventKind.TOOL_CALL_CLAIMED,
                activation_id="act-1",
                step_index=2,
                correlation_id="call-1",
                payload={"tool_name": "lookup", "允许": True},
            )
        ]
    )
    _patch_runner(monkeypatch, runner)
    renderer, stdout, stderr = _renderer(json_output=False)

    code = run_events(
        RunEventsOptions(config_path=Path("agent.yaml"), run_id="run-1"),
        renderer=renderer,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    rendered = stdout.getvalue()
    assert rendered.count("Iris Run Events") == 1
    assert "sequence: 3" in rendered
    assert "kind: tool_call.claimed" in rendered
    assert "occurred_at: 2026-01-01T00:00:00Z" in rendered
    assert "session_id: session-1" in rendered
    assert "activation_id: act-1" in rendered
    assert "step_index: 2" in rendered
    assert "correlation_id: call-1" in rendered
    assert 'payload: {"tool_name":"lookup","允许":true}' in rendered


def test_run_events_empty_result_keeps_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _RecordingEventsRunner()
    _patch_runner(monkeypatch, runner)
    renderer, stdout, stderr = _renderer(json_output=True)

    code = run_events(
        RunEventsOptions(
            config_path=Path("agent.yaml"),
            run_id="run-1",
            after_sequence=12,
            json_output=True,
        ),
        renderer=renderer,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["events"] == []
    assert payload["next_after_sequence"] == 12


def test_run_events_human_empty_result_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _RecordingEventsRunner()
    _patch_runner(monkeypatch, runner)
    renderer, stdout, stderr = _renderer(json_output=False)

    code = run_events(
        RunEventsOptions(
            config_path=Path("agent.yaml"),
            run_id="run-1",
            after_sequence=12,
        ),
        renderer=renderer,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    rendered = stdout.getvalue()
    assert rendered.count("Iris Run Events") == 1
    assert "event_count: 0" in rendered
    assert "next_after_sequence: 12" in rendered
    assert "events: 无" in rendered


def test_run_events_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _RecordingEventsRunner(error=KeyboardInterrupt())
    _patch_runner(monkeypatch, runner)
    renderer, stdout, stderr = _renderer(json_output=True)

    code = run_events(
        RunEventsOptions(config_path=Path("agent.yaml"), run_id="run-1"),
        renderer=renderer,
    )

    assert code == 130
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload["error"]["code"] == "INTERRUPTED"
