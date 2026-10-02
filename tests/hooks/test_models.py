"""Hooks 公共输入边界与真实领域模型的事件投影。"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.hooks import (
    HookRegistration,
    RunFinishedEvent,
    RunStartedEvent,
    ToolAfterEvent,
    ToolAfterResult,
    ToolBeforeEvent,
    ToolBeforeResult,
    event_to_dict,
)
from iris.lifecycle import RunLimits, RunPhase, RunResult, RunSnapshot, RunStopReason, RunUsage
from iris.message import DataBlock, ImageBlock, ImageFileRef, TextBlock
from iris.tools import ToolResult


def _run(terminal: bool = False) -> RunSnapshot:
    now = datetime.now(UTC)
    return RunSnapshot(
        run_id="run",
        session_id="session",
        agent_id="agent",
        phase=RunPhase.TERMINAL if terminal else RunPhase.ACTIVE,
        stop_reason=RunStopReason.COMPLETED if terminal else None,
        revision=1,
        current_activation_id=None if terminal else "activation",
        limits=RunLimits(),
        usage=RunUsage(),
        checkpoint_sequence=0,
        last_event_sequence=1,
        created_at=now,
        started_at=now,
        updated_at=now,
        finished_at=now if terminal else None,
    )


@pytest.mark.parametrize("with_image", [False, True])
def test_four_events_serialize_real_models(with_image: bool) -> None:
    ref = ImageFileRef(path=Path.cwd() / "image.png", mime_type="image/png", width=1, height=1)
    content: str | list[DataBlock] = (
        [TextBlock(text="hello"), ImageBlock(original=ref, model=ref)] if with_image else "hello"
    )
    common = dict(
        agent_id="agent",
        session_id="session",
        run_id="run",
        activation_id="activation",
        workspace="J:/workspace",
    )
    events = [
        RunStartedEvent(**common, run=_run(), input=content),
        RunFinishedEvent(**common, result=RunResult(run=_run(True))),
        ToolBeforeEvent(**common, call_id="call", tool_name="exec_command", arguments={"x": 1}),
        ToolAfterEvent(
            **common,
            call_id="call",
            tool_name="exec_command",
            arguments={"x": 1},
            result=ToolResult(
                tool_use_id="call", tool_name="exec_command", content=[TextBlock(text="done")]
            ),
            body_status="success",
        ),
    ]
    payloads = [json.loads(json.dumps(event_to_dict(event))) for event in events]
    assert [payload["event"] for payload in payloads] == [
        "run.started",
        "run.finished",
        "tool.before",
        "tool.after",
    ]
    assert payloads[0]["run"]["phase"] == "active"
    assert payloads[0]["input"] == (
        [block.model_dump(mode="json") for block in content] if with_image else content
    )
    assert payloads[1]["result"]["run"]["stop_reason"] == "completed"
    assert payloads[2]["tool_name"] == "exec_command"
    assert payloads[3]["result"]["content"][0]["text"] == "done"
    assert all(event.occurred_at.tzinfo is UTC for event in events)
    with pytest.raises(FrozenInstanceError):
        events[0].agent_id = "changed"


@pytest.mark.parametrize(
    "model, field", [(ToolBeforeResult, "deny_reason"), (ToolAfterResult, "feedback")]
)
def test_hook_results_validate_only_public_shape(model: type, field: str) -> None:
    valid = model(**{field: "reason"})
    assert getattr(valid, field) == "reason"
    for text in ["", "  "]:
        with pytest.raises(ValidationError):
            model(**{field: text})
    with pytest.raises(ValidationError):
        model(**{field: "reason", "unexpected": 1})
    with pytest.raises(ValidationError):
        setattr(valid, field, "changed")


def test_registration_has_one_validated_sdk_boundary() -> None:
    async def handler(event: ToolBeforeEvent) -> None:
        return None

    registration = HookRegistration(
        event="tool.before", name="audit", handler=handler, tool_names=["exec_command"]
    )
    assert registration.timeout_seconds == 10
    assert registration.tool_names == ("exec_command",)
    for override in [
        {"event": "other"},
        {"timeout_seconds": 0},
        {"tool_names": []},
        {"event": "run.started", "tool_names": ["exec_command"]},
        {"handler": None},
    ]:
        with pytest.raises(ValidationError):
            HookRegistration(**dict(event="tool.before", name="audit", handler=handler) | override)
