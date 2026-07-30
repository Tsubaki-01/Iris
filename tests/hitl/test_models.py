from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
    make_call_fingerprint,
)


def _call() -> ToolCallSnapshot:
    fingerprint = make_call_fingerprint(
        session_id="session-1",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="write",
        arguments={"value": "x"},
        workspace_root="/workspace",
    )
    return ToolCallSnapshot(
        tool_call_id="call-1",
        tool_name="write",
        arguments={"value": "x"},
        workspace_root="/workspace",
        fingerprint=fingerprint,
    )


def _interaction(**updates: object) -> HumanInteraction:
    values: dict[str, object] = {
        "interaction_id": "interaction-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "step_index": 0,
        "tool_call_id": "call-1",
        "request": HumanInteractionRequest(
            tool_call=_call(),
            prompt=PermissionPrompt(reason="需要写入"),
        ),
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(updates)
    return HumanInteraction.model_validate(values)


def test_call_fingerprint_is_stable_and_subject_sensitive() -> None:
    first = _call()
    second = _call()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    changed = make_call_fingerprint(
        session_id="session-1",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="write",
        arguments={"value": "y"},
        workspace_root="/workspace",
    )
    assert changed != first.fingerprint


def test_interaction_requires_exact_tool_subject() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        _interaction(tool_call_id="other")


def test_pending_resolved_and_closed_facts_are_consistent() -> None:
    pending = _interaction()
    response = PermissionInteractionResponse(decision="approve")
    resolved = _interaction(
        status=InteractionStatus.RESOLVED,
        response=response,
        resolved_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    closed = resolved.model_copy(
        update={
            "status": InteractionStatus.CLOSED,
            "closed_at": datetime(2026, 1, 3, tzinfo=UTC),
            "close_reason": "completed",
        }
    )

    assert pending.status is InteractionStatus.PENDING
    assert resolved.response == response
    assert HumanInteraction.model_validate(closed.model_dump()) == closed


def test_interaction_rejects_missing_response_or_close_facts() -> None:
    with pytest.raises(ValidationError, match="response"):
        _interaction(status=InteractionStatus.RESOLVED)
    with pytest.raises(ValidationError, match="closed"):
        _interaction(status=InteractionStatus.CLOSED)


def test_prompt_and_response_kinds_must_match() -> None:
    with pytest.raises(ValidationError, match="kind"):
        _interaction(
            status=InteractionStatus.RESOLVED,
            response=QuestionInteractionResponse(answer="不匹配"),
        )


def test_question_prompt_rejects_duplicate_options() -> None:
    with pytest.raises(ValidationError, match="重复"):
        QuestionPrompt(question="选择？", options=["a", "a"])
