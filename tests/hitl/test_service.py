from __future__ import annotations

import pytest

from iris.exceptions import (
    HITLAlreadyConsumedError,
    HITLCheckpointInvalidError,
    HITLConflictError,
    HITLNotFoundError,
    HITLResponseMismatchError,
)
from iris.hitl import (
    HumanInteraction,
    HumanInteractionResponse,
    HumanInteractionService,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)


def test_service_creates_permission_with_a_bound_fingerprint() -> None:
    store = _FakeStore()
    interaction = _service(store).create_permission(
        session_id="session_1",
        run_id="run_1",
        step_index=0,
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        reason="needs approval",
        workspace_root="C:/workspace",
        checkpoint={"checkpoint_version": 1},
    )

    assert interaction.status is InteractionStatus.PENDING
    assert interaction.request.call_fingerprint
    assert store.interactions[interaction.interaction_id] == interaction


def test_resolving_same_response_twice_is_idempotent() -> None:
    store = _FakeStore()
    service = _service(store)
    interaction = _permission_interaction(service)
    response = PermissionInteractionResponse(decision="approve")

    resolved = service.resolve(interaction.interaction_id, response)
    repeated = service.resolve(interaction.interaction_id, response)

    assert repeated == resolved
    assert store.resolve_calls == 1


def test_resolving_different_response_conflicts() -> None:
    service = _service(_FakeStore())
    interaction = _permission_interaction(service)
    service.resolve(interaction.interaction_id, PermissionInteractionResponse(decision="approve"))

    with pytest.raises(HITLConflictError):
        service.resolve(
            interaction.interaction_id, PermissionInteractionResponse(decision="reject")
        )


def test_resolving_a_response_of_another_kind_is_rejected() -> None:
    service = _service(_FakeStore())
    interaction = _permission_interaction(service)

    with pytest.raises(HITLResponseMismatchError):
        service.resolve(interaction.interaction_id, QuestionInteractionResponse(answer="继续"))


def test_service_reports_missing_interactions() -> None:
    with pytest.raises(HITLNotFoundError):
        _service(_FakeStore()).get("int_missing")


def test_claim_and_update_reject_invalid_checkpoint_values() -> None:
    service = _service(_FakeStore())
    interaction = _permission_interaction(service)
    service.resolve(interaction.interaction_id, PermissionInteractionResponse(decision="approve"))

    with pytest.raises(HITLCheckpointInvalidError):
        service.claim(interaction.interaction_id, {"not_json": {"set"}})


def test_claimed_interaction_cannot_be_resolved_again() -> None:
    service = _service(_FakeStore())
    interaction = _permission_interaction(service)
    service.resolve(interaction.interaction_id, PermissionInteractionResponse(decision="approve"))
    service.claim(interaction.interaction_id, {"checkpoint_version": 1})

    with pytest.raises(HITLAlreadyConsumedError):
        service.resolve(
            interaction.interaction_id, PermissionInteractionResponse(decision="approve")
        )


class _FakeStore:
    def __init__(self) -> None:
        self.interactions: dict[str, HumanInteraction] = {}
        self.resolve_calls = 0

    def create_interaction(self, interaction: HumanInteraction) -> None:
        self.interactions[interaction.interaction_id] = interaction

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        return self.interactions.get(interaction_id)

    def list_pending_interactions(self, session_id: str | None = None) -> list[HumanInteraction]:
        return [
            interaction
            for interaction in self.interactions.values()
            if interaction.status is InteractionStatus.PENDING
            and (session_id is None or interaction.session_id == session_id)
        ]

    def resolve_interaction(
        self,
        interaction_id: str,
        response: HumanInteractionResponse,
        *,
        expected_version: int,
    ) -> HumanInteraction:
        interaction = self.interactions[interaction_id]
        assert interaction.version == expected_version
        self.resolve_calls += 1
        resolved = interaction.model_copy(
            update={
                "status": InteractionStatus.RESOLVED,
                "response": response,
                "version": expected_version + 1,
            }
        )
        self.interactions[interaction_id] = resolved
        return resolved

    def claim_interaction(
        self,
        interaction_id: str,
        checkpoint: dict[str, object],
        *,
        expected_version: int,
    ) -> HumanInteraction:
        interaction = self.interactions[interaction_id]
        assert interaction.version == expected_version
        claimed = interaction.model_copy(
            update={
                "status": InteractionStatus.CONSUMED,
                "resume_phase": InteractionResumePhase.CLAIMED,
                "checkpoint": checkpoint,
                "version": expected_version + 1,
            }
        )
        self.interactions[interaction_id] = claimed
        return claimed

    def update_consumed_interaction(
        self,
        interaction_id: str,
        *,
        resume_phase: InteractionResumePhase,
        checkpoint: dict[str, object],
        expected_version: int,
    ) -> HumanInteraction:
        interaction = self.interactions[interaction_id]
        assert interaction.version == expected_version
        updated = interaction.model_copy(
            update={
                "resume_phase": resume_phase,
                "checkpoint": checkpoint,
                "version": expected_version + 1,
            }
        )
        self.interactions[interaction_id] = updated
        return updated


def _service(store: _FakeStore) -> HumanInteractionService:
    return HumanInteractionService(store)


def _permission_interaction(service: HumanInteractionService) -> HumanInteraction:
    return service.create_permission(
        session_id="session_1",
        run_id="run_1",
        step_index=0,
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        reason="needs approval",
        workspace_root="C:/workspace",
        checkpoint={"checkpoint_version": 1},
    )
