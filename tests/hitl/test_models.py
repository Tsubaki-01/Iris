from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from iris.hitl import (
    HumanInteraction,
    InteractionKind,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionRequest,
    PermissionInteractionResponse,
    QuestionInteractionRequest,
    QuestionInteractionResponse,
    make_call_fingerprint,
)
from iris.hitl.models import HumanInteractionRequest, HumanInteractionResponse


def test_permission_interaction_round_trips_as_json() -> None:
    interaction = _permission_interaction()

    restored = HumanInteraction.model_validate_json(interaction.model_dump_json())

    assert restored == interaction
    assert isinstance(restored.request, PermissionInteractionRequest)
    assert isinstance(restored.response, PermissionInteractionResponse)


def test_question_request_and_response_round_trip_as_discriminated_unions() -> None:
    request = TypeAdapter(HumanInteractionRequest).validate_json(
        QuestionInteractionRequest(
            tool_call_id="call_question",
            question="继续执行吗？",
            options=["继续", "取消"],
        ).model_dump_json()
    )
    response = TypeAdapter(HumanInteractionResponse).validate_json(
        QuestionInteractionResponse(answer="继续").model_dump_json()
    )

    assert isinstance(request, QuestionInteractionRequest)
    assert isinstance(response, QuestionInteractionResponse)


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (PermissionInteractionRequest, "reason", "  "),
        (QuestionInteractionRequest, "question", "  "),
        (QuestionInteractionRequest, "options", ["继续", "  "]),
        (QuestionInteractionRequest, "options", ["继续", " 继续 "]),
        (QuestionInteractionResponse, "answer", "  "),
    ],
)
def test_request_and_response_reject_empty_or_duplicate_human_text(
    factory: type[object], field: str, value: object
) -> None:
    common = {
        "tool_call_id": "call_1",
        "tool_name": "write_file",
        "arguments": {},
        "reason": "needs approval",
        "workspace_root": "C:/workspace",
        "call_fingerprint": "a" * 64,
        "question": "继续吗？",
        "answer": "继续",
    }
    kwargs = {key: item for key, item in common.items() if key in factory.model_fields}
    kwargs[field] = value

    with pytest.raises(ValidationError):
        factory(**kwargs)


@pytest.mark.parametrize("field", ["interaction_id", "session_id", "run_id", "tool_call_id"])
def test_human_interaction_rejects_blank_ids(field: str) -> None:
    values = _permission_interaction().model_dump()
    values[field] = "  "

    with pytest.raises(ValidationError):
        HumanInteraction.model_validate(values)


def test_human_interaction_rejects_request_response_kind_mismatch() -> None:
    values = _permission_interaction().model_dump()
    values["response"] = {"kind": "question", "answer": "继续"}

    with pytest.raises(ValidationError):
        HumanInteraction.model_validate(values)


@pytest.mark.parametrize(
    ("status", "resume_phase", "response"),
    [
        (
            InteractionStatus.PENDING,
            InteractionResumePhase.WAITING,
            {"kind": "permission", "decision": "approve"},
        ),
        (InteractionStatus.RESOLVED, InteractionResumePhase.WAITING, None),
        (
            InteractionStatus.CONSUMED,
            InteractionResumePhase.WAITING,
            {"kind": "permission", "decision": "approve"},
        ),
        (
            InteractionStatus.RESOLVED,
            InteractionResumePhase.CLAIMED,
            {"kind": "permission", "decision": "approve"},
        ),
    ],
)
def test_human_interaction_enforces_status_and_resume_phase_invariants(
    status: InteractionStatus,
    resume_phase: InteractionResumePhase,
    response: dict[str, str] | None,
) -> None:
    values = _permission_interaction().model_dump()
    values.update(status=status, resume_phase=resume_phase, response=response)

    with pytest.raises(ValidationError):
        HumanInteraction.model_validate(values)


def test_human_interaction_rejects_non_json_safe_checkpoint_and_arguments() -> None:
    interaction = _permission_interaction()
    values = interaction.model_dump()
    values["checkpoint"] = {"created_at": datetime.now(UTC)}

    with pytest.raises(ValidationError):
        HumanInteraction.model_validate(values)
    with pytest.raises(ValidationError):
        PermissionInteractionRequest(
            tool_call_id="call_1",
            tool_name="write_file",
            arguments={"not_json": {1, 2}},
            reason="needs approval",
            workspace_root="C:/workspace",
            call_fingerprint="a" * 64,
        )


def test_call_fingerprint_is_canonical_and_binds_call_workspace_and_arguments() -> None:
    fingerprint = make_call_fingerprint(
        session_id="session_1",
        run_id="run_1",
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"path": "notes.txt", "content": "hello"},
        workspace_root="C:/workspace",
    )

    assert fingerprint == make_call_fingerprint(
        session_id="session_1",
        run_id="run_1",
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"content": "hello", "path": "notes.txt"},
        workspace_root="C:/workspace",
    )
    assert fingerprint != make_call_fingerprint(
        session_id="session_1",
        run_id="run_1",
        tool_call_id="call_2",
        tool_name="write_file",
        arguments={"path": "notes.txt", "content": "hello"},
        workspace_root="C:/workspace",
    )
    assert fingerprint != make_call_fingerprint(
        session_id="session_1",
        run_id="run_1",
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"path": "other.txt", "content": "hello"},
        workspace_root="C:/workspace",
    )
    assert fingerprint != make_call_fingerprint(
        session_id="session_1",
        run_id="run_1",
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"path": "notes.txt", "content": "hello"},
        workspace_root="C:/other-workspace",
    )


def _permission_interaction() -> HumanInteraction:
    return HumanInteraction(
        interaction_id="int_0123456789abcdef0123456789abcdef",
        session_id="session_1",
        run_id="run_1",
        step_index=0,
        tool_call_id="call_1",
        kind=InteractionKind.PERMISSION,
        status=InteractionStatus.RESOLVED,
        resume_phase=InteractionResumePhase.WAITING,
        request=PermissionInteractionRequest(
            tool_call_id="call_1",
            tool_name="write_file",
            arguments={"path": "notes.txt"},
            reason="needs approval",
            workspace_root="C:/workspace",
            call_fingerprint="a" * 64,
        ),
        response=PermissionInteractionResponse(decision="approve"),
        checkpoint={"checkpoint_version": 1},
    )
