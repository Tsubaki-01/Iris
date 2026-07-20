from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from iris.exceptions import HITLConflictError
from iris.hitl import (
    HumanInteraction,
    InMemoryInteractionStore,
    InteractionKind,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionRequest,
    PermissionInteractionResponse,
)
from iris.store import SQLiteStore


class _StoreFactory(Protocol):
    def __call__(self, path: Path) -> object: ...


@pytest.fixture(params=["memory", "sqlite"])
def interaction_store(request: pytest.FixtureRequest, tmp_path: Path) -> object:
    factories: dict[str, _StoreFactory] = {
        "memory": lambda _path: InMemoryInteractionStore(),
        "sqlite": lambda path: SQLiteStore(path),
    }
    return factories[request.param](tmp_path / "interactions.db")


def test_store_round_trips_json_interaction(interaction_store: object) -> None:
    store = _as_store(interaction_store)
    interaction = _interaction(session_id="session_1", interaction_id="int_" + "1" * 32)

    store.create_interaction(interaction)

    assert store.load_interaction(interaction.interaction_id) == interaction


def test_store_lists_only_pending_interactions_in_creation_order(interaction_store: object) -> None:
    store = _as_store(interaction_store)
    first = _interaction(session_id="session_1", interaction_id="int_" + "1" * 32)
    second = _interaction(session_id="session_2", interaction_id="int_" + "2" * 32)
    store.create_interaction(first)
    store.create_interaction(second)
    resolved = store.resolve_interaction(
        second.interaction_id,
        PermissionInteractionResponse(decision="approve"),
        expected_version=second.version,
    )

    assert [item.interaction_id for item in store.list_pending_interactions()] == [
        first.interaction_id
    ]
    assert store.list_pending_interactions("session_2") == []
    assert resolved.status is InteractionStatus.RESOLVED


def test_store_allows_one_active_interaction_per_session(interaction_store: object) -> None:
    store = _as_store(interaction_store)
    store.create_interaction(_interaction(session_id="session_1", interaction_id="int_" + "1" * 32))

    with pytest.raises(HITLConflictError):
        store.create_interaction(
            _interaction(session_id="session_1", interaction_id="int_" + "2" * 32)
        )


def test_result_committed_interaction_no_longer_blocks_its_session(
    interaction_store: object,
) -> None:
    store = _as_store(interaction_store)
    interaction = _interaction(session_id="session_1", interaction_id="int_" + "1" * 32)
    store.create_interaction(interaction)
    resolved = store.resolve_interaction(
        interaction.interaction_id,
        PermissionInteractionResponse(decision="approve"),
        expected_version=interaction.version,
    )
    claimed = store.claim_interaction(
        interaction.interaction_id,
        {"checkpoint_version": 1, "result": "ready"},
        expected_version=resolved.version,
    )
    store.update_consumed_interaction(
        interaction.interaction_id,
        resume_phase=InteractionResumePhase.RESULT_COMMITTED,
        checkpoint={"checkpoint_version": 1, "result": "committed"},
        expected_version=claimed.version,
    )

    store.create_interaction(_interaction(session_id="session_1", interaction_id="int_" + "2" * 32))


def test_store_rejects_stale_compare_and_set_versions(interaction_store: object) -> None:
    store = _as_store(interaction_store)
    interaction = _interaction(session_id="session_1", interaction_id="int_" + "1" * 32)
    store.create_interaction(interaction)

    with pytest.raises(HITLConflictError):
        store.resolve_interaction(
            interaction.interaction_id,
            PermissionInteractionResponse(decision="approve"),
            expected_version=interaction.version + 1,
        )
    resolved = store.resolve_interaction(
        interaction.interaction_id,
        PermissionInteractionResponse(decision="approve"),
        expected_version=interaction.version,
    )
    with pytest.raises(HITLConflictError):
        store.claim_interaction(
            interaction.interaction_id,
            {"checkpoint_version": 1},
            expected_version=interaction.version,
        )
    claimed = store.claim_interaction(
        interaction.interaction_id,
        {"checkpoint_version": 1},
        expected_version=resolved.version,
    )
    with pytest.raises(HITLConflictError):
        store.update_consumed_interaction(
            interaction.interaction_id,
            resume_phase=InteractionResumePhase.RESULT_READY,
            checkpoint={"checkpoint_version": 1},
            expected_version=resolved.version,
        )
    updated = store.update_consumed_interaction(
        interaction.interaction_id,
        resume_phase=InteractionResumePhase.RESULT_READY,
        checkpoint={"checkpoint_version": 1, "result": "ready"},
        expected_version=claimed.version,
    )

    assert updated.resume_phase is InteractionResumePhase.RESULT_READY


def test_sqlite_store_persists_interactions_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "interactions.db"
    interaction = _interaction(session_id="session_1", interaction_id="int_" + "1" * 32)
    SQLiteStore(path).create_interaction(interaction)

    loaded = SQLiteStore(path).load_interaction(interaction.interaction_id)

    assert loaded == interaction


def _as_store(store: object) -> _InteractionStore:
    assert isinstance(store, _InteractionStore)
    return store


@runtime_checkable
class _InteractionStore(Protocol):
    def create_interaction(self, interaction: HumanInteraction) -> None: ...

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None: ...

    def list_pending_interactions(
        self, session_id: str | None = None
    ) -> list[HumanInteraction]: ...

    def resolve_interaction(
        self,
        interaction_id: str,
        response: PermissionInteractionResponse,
        *,
        expected_version: int,
    ) -> HumanInteraction: ...

    def claim_interaction(
        self,
        interaction_id: str,
        checkpoint: dict[str, object],
        *,
        expected_version: int,
    ) -> HumanInteraction: ...

    def update_consumed_interaction(
        self,
        interaction_id: str,
        *,
        resume_phase: InteractionResumePhase,
        checkpoint: dict[str, object],
        expected_version: int,
    ) -> HumanInteraction: ...


def _interaction(*, session_id: str, interaction_id: str) -> HumanInteraction:
    return HumanInteraction(
        interaction_id=interaction_id,
        session_id=session_id,
        run_id="run_1",
        step_index=0,
        tool_call_id="call_1",
        kind=InteractionKind.PERMISSION,
        request=PermissionInteractionRequest(
            tool_call_id="call_1",
            tool_name="write_file",
            arguments={"path": "notes.txt", "content": "hello"},
            reason="needs approval",
            workspace_root="C:/workspace",
            call_fingerprint="a" * 64,
        ),
        checkpoint={"checkpoint_version": 1, "metadata": {"source": "test"}},
    )
