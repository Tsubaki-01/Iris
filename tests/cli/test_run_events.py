from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import iris.cli.run as run_module
from iris.cli.run import RunEventsOptions, run_events
from iris.cli.run_output import RunCommandRenderer
from iris.lifecycle import RunEvent, RunEventKind


class _RecordingEventsRunner:
    def __init__(self, events: list[RunEvent]) -> None:
        self.events = events
        self.calls: list[tuple[str, int]] = []

    def list_events(self, run_id: str, *, after_sequence: int = 0) -> list[RunEvent]:
        self.calls.append((run_id, after_sequence))
        return self.events


def test_run_events_calls_public_runner_once_and_emits_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        RunEvent(
            run_id="run-1",
            session_id="session-1",
            sequence=8,
            kind=RunEventKind.ACTIVATION_STARTED,
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        RunEvent(
            run_id="run-1",
            session_id="session-1",
            sequence=9,
            kind=RunEventKind.MODEL_STEP_RESERVED,
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    ]
    runner = _RecordingEventsRunner(events)
    monkeypatch.setattr(run_module, "_build_non_executing_runner", lambda options: runner)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_events(
        RunEventsOptions(
            config_path=Path("agent.yaml"),
            run_id=" run-1 ",
            after_sequence=7,
            json_output=True,
        ),
        renderer=RunCommandRenderer(json_output=True, stdout=stdout, stderr=stderr),
    )

    assert code == 0
    assert runner.calls == [("run-1", 7)]
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["next_after_sequence"] == 9
    assert [event["sequence"] for event in payload["events"]] == [8, 9]
