"""无状态 HITL 领域服务契约。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iris.exceptions import HITLResponseMismatchError, IrisRunRecoveryError, IrisRunStateError
from iris.hitl import (
    ApprovedToolCall,
    HumanInteractionRequest,
    HumanInteractionService,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import RunLimits, RunPhase, RunSnapshot, RunUsage
from iris.tools import ToolResult

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_FINGERPRINT = "a" * 64


def test_service_is_zero_argument_and_has_no_store_state() -> None:
    service = HumanInteractionService()

    assert not hasattr(service, "store")
    assert vars(service) == {}


@pytest.mark.parametrize(
    "prompt", [QuestionPrompt(question="继续吗？"), PermissionPrompt(reason="写入")]
)
def test_create_pending_binds_request_to_active_run_without_side_effects(prompt: object) -> None:
    service = HumanInteractionService()
    run = _run(phase=RunPhase.ACTIVE, current_activation_id="act-1")
    request = _request(prompt=prompt)
    expires_at = _NOW + timedelta(minutes=5)

    interaction = service.create_pending(
        request,
        run=run,
        step_index=3,
        expires_at=expires_at,
    )

    assert interaction.run_id == run.run_id
    assert interaction.session_id == run.session_id
    assert interaction.tool_call_id == request.tool_call.tool_call_id
    assert interaction.step_index == 3
    assert interaction.status is InteractionStatus.PENDING
    assert interaction.version == 1
    assert interaction.created_at == run.updated_at
    assert interaction.expires_at == expires_at


def test_create_pending_rejects_non_active_run_and_naive_expiry() -> None:
    service = HumanInteractionService()
    with pytest.raises(IrisRunStateError):
        service.create_pending(
            _request(),
            run=_run(phase=RunPhase.WAITING, pending_interaction_id="int-1"),
            step_index=0,
            expires_at=None,
        )
    with pytest.raises(IrisRunStateError):
        service.create_pending(
            _request(),
            run=_run(phase=RunPhase.ACTIVE, current_activation_id="act-1"),
            step_index=0,
            expires_at=datetime(2026, 1, 2),
        )


def test_validate_response_checks_identity_kind_expiry_and_environment() -> None:
    service = HumanInteractionService()
    interaction = service.create_pending(
        _request(),
        run=_run(phase=RunPhase.ACTIVE, current_activation_id="act-1"),
        step_index=0,
        expires_at=_NOW + timedelta(minutes=1),
    )
    waiting = _run(
        phase=RunPhase.WAITING,
        pending_interaction_id=interaction.interaction_id,
        revision=2,
    )

    service.validate_response(
        interaction,
        run=waiting,
        response=QuestionInteractionResponse(answer="继续"),
        now=_NOW,
        environment_fingerprint=_FINGERPRINT,
    )
    with pytest.raises(HITLResponseMismatchError):
        service.validate_response(
            interaction,
            run=waiting,
            response=PermissionInteractionResponse(decision="approve"),
            now=_NOW,
            environment_fingerprint=_FINGERPRINT,
        )
    with pytest.raises(IrisRunRecoveryError):
        service.validate_response(
            interaction,
            run=waiting,
            response=QuestionInteractionResponse(answer="继续"),
            now=_NOW,
            environment_fingerprint="environment-v2",
        )
    with pytest.raises(IrisRunStateError):
        service.validate_response(
            interaction,
            run=waiting,
            response=QuestionInteractionResponse(answer="继续"),
            now=interaction.expires_at,
            environment_fingerprint=_FINGERPRINT,
        )


def test_project_response_returns_question_rejection_and_approval_values() -> None:
    service = HumanInteractionService()
    active = _run(phase=RunPhase.ACTIVE, current_activation_id="act-1")

    question = service.create_pending(_request(), run=active, step_index=0, expires_at=None)
    question_response = QuestionInteractionResponse(answer="继续")
    question_resolved = question.model_copy(
        update={
            "status": InteractionStatus.RESOLVED,
            "response": question_response,
            "version": 2,
            "resolved_at": _NOW,
        }
    )
    answer = service.project_response(question_resolved, question_response)
    assert isinstance(answer, ToolResult)
    assert answer.tool_use_id == "call-1"
    assert answer.model_content == "继续"

    permission = service.create_pending(
        _request(prompt=PermissionPrompt(reason="写入")),
        run=active,
        step_index=0,
        expires_at=None,
    )
    rejected_response = PermissionInteractionResponse(decision="reject")
    rejected = service.project_response(
        permission.model_copy(
            update={
                "status": InteractionStatus.RESOLVED,
                "response": rejected_response,
                "version": 2,
                "resolved_at": _NOW,
            }
        ),
        rejected_response,
    )
    assert isinstance(rejected, ToolResult)
    assert rejected.is_error is True
    assert rejected.error is not None and rejected.error.code == "USER_REJECTED"

    approved_response = PermissionInteractionResponse(decision="approve")
    approved = service.project_response(
        permission.model_copy(
            update={
                "status": InteractionStatus.RESOLVED,
                "response": approved_response,
                "version": 2,
                "resolved_at": _NOW,
            }
        ),
        approved_response,
    )
    assert approved == ApprovedToolCall(
        interaction_id=permission.interaction_id,
        tool_call_id="call-1",
        tool_name="ask_question",
        fingerprint=_FINGERPRINT,
    )


def _request(*, prompt: object | None = None) -> HumanInteractionRequest:
    return HumanInteractionRequest(
        tool_call=ToolCallSnapshot(
            tool_call_id="call-1",
            tool_name="ask_question",
            arguments={"question": "继续吗？"},
            workspace_root="workspace",
            fingerprint=_FINGERPRINT,
        ),
        prompt=prompt or QuestionPrompt(question="继续吗？"),
    )


def _run(
    *,
    phase: RunPhase,
    current_activation_id: str | None = None,
    pending_interaction_id: str | None = None,
    revision: int = 1,
) -> RunSnapshot:
    return RunSnapshot(
        run_id="run-1",
        session_id="session-1",
        agent_id="agent-1",
        phase=phase,
        revision=revision,
        current_activation_id=current_activation_id,
        pending_interaction_id=pending_interaction_id,
        limits=RunLimits(),
        usage=RunUsage(),
        environment_fingerprint=_FINGERPRINT,
        checkpoint_sequence=1,
        last_event_sequence=1,
        created_at=_NOW,
        started_at=_NOW,
        updated_at=_NOW,
    )
