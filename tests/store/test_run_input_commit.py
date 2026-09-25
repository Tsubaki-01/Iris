"""Run 输入提交必须独立于模型请求原子保存。"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import iris.store.sqlite as sqlite_module
from iris.exceptions import IrisRunConflictError, IrisRunPersistenceError, IrisRunStateError
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CommitRunInput,
    CreateRun,
    FinishRun,
    ForkSession,
    LifecycleStore,
    MemoryOverviewSource,
    RequestCancellation,
    ReserveModelStep,
    RunCheckpoint,
    RunStopReason,
    SessionContextWindow,
)
from iris.message import Msg
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .test_lifecycle_store_contract import _compaction_command

NOW = datetime(2026, 9, 19, tzinfo=UTC)


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """同一契约覆盖两种 store。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "input.db")
    return InMemoryLifecycleStore()


def _window(revision: int = 1) -> SessionContextWindow:
    return SessionContextWindow(
        memory_overview=f"memory overview {revision}",
        mode="full",
        sources=(
            MemoryOverviewSource(
                namespace="project", path="project/overview.md", source_revision=revision
            ),
        ),
    )


def _input_command(
    store: LifecycleStore, *, run_id: str = "run-input", session_id: str = "session"
) -> CommitRunInput:
    session = store.load_session(session_id)
    activation_id = f"{run_id}-activation"
    checkpoint = RunCheckpoint(
        run_id=run_id,
        sequence=1,
        activation_id=activation_id,
        engine_cursor={"position": "before_input", "step_index": 0, "visible_tool_names": []},
        session_revision=session.revision,
        model_steps_reserved=0,
        model_steps_committed=0,
    )
    created = store.create_run(
        CreateRun(
            request=AgentRunRequest(input="question", session_id=session_id, run_id=run_id),
            options=AgentRunOptions(),
            agent_id="agent",
            start_activation_id=activation_id,
            initial_checkpoint=checkpoint,
            now=NOW,
        )
    )
    return CommitRunInput(
        run_id=created.run.run_id,
        expected_run_revision=created.run.revision,
        activation_id=activation_id,
        expected_session_revision=session.revision,
        message_delta=[Msg.user("BCI"), Msg.user("question")],
        initial_context_window=_window(),
        checkpoint=checkpoint.model_copy(
            update={
                "sequence": 2,
                "session_revision": session.revision + 1,
                "engine_cursor": {
                    "position": "before_model",
                    "step_index": 0,
                    "visible_tool_names": [],
                },
            }
        ),
        now=NOW,
    )


def test_input_commit_saves_history_and_cursor_without_model_step(store: LifecycleStore) -> None:
    command = _input_command(store)
    before = store.load_run(command.run_id)
    events = store.list_events(command.run_id)

    committed = store.commit_run_input(command)

    assert before is not None
    assert committed.run.revision == before.revision + 1
    assert committed.run.usage == before.usage
    assert committed.run.last_event_sequence == before.last_event_sequence
    assert committed.events == ()
    assert committed.session_revision == 1
    assert committed.checkpoint == command.checkpoint
    assert committed.checkpoint.checkpoint_version == 3
    assert committed.run.checkpoint_sequence == 2
    assert store.load_checkpoint(command.run_id) == command.checkpoint
    assert store.load_session("session").messages == command.message_delta
    assert store.load_session("session").context_window == command.initial_context_window
    assert store.list_events(command.run_id) == events
    if isinstance(store, SQLiteStore):
        assert SQLiteStore(store.path).load_session("session").context_window == _window()
    with pytest.raises(IrisRunConflictError):
        store.commit_run_input(command)
    with pytest.raises(IrisRunStateError, match="before_input"):
        store.commit_run_input(
            replace(
                command,
                expected_run_revision=committed.run.revision,
                expected_session_revision=1,
            )
        )
    assert store.load_session("session").messages == command.message_delta


@pytest.mark.parametrize("mismatch", ["session", "fence", "position", "step", "counter"])
def test_input_commit_rejects_changed_facts_atomically(
    store: LifecycleStore, mismatch: str
) -> None:
    command = _input_command(store)
    before = store.load_run(command.run_id)
    checkpoint_before = store.load_checkpoint(command.run_id)
    if mismatch == "session":
        command = replace(command, expected_session_revision=1)
    elif mismatch == "fence":
        command = replace(command, activation_id="stale")
    else:
        changes: dict[str, object]
        if mismatch == "counter":
            changes = {"model_steps_reserved": 1}
        else:
            changes = {
                "engine_cursor": {
                    "position": "tool_batch" if mismatch == "position" else "before_model",
                    "step_index": 1 if mismatch == "step" else 0,
                    "visible_tool_names": [],
                }
            }
        command = replace(command, checkpoint=command.checkpoint.model_copy(update=changes))

    with pytest.raises((IrisRunConflictError, IrisRunStateError)):
        store.commit_run_input(command)

    assert store.load_run(command.run_id) == before
    assert store.load_checkpoint(command.run_id) == checkpoint_before
    assert store.load_session("session").messages == []
    assert store.load_session("session").context_window is None


def test_input_requires_explicit_initial_window(store: LifecycleStore) -> None:
    command = replace(_input_command(store), initial_context_window=None)
    before = store.load_run(command.run_id)
    checkpoint = store.load_checkpoint(command.run_id)
    with pytest.raises(IrisRunStateError, match="context window"):
        store.commit_run_input(command)
    assert store.load_run(command.run_id) == before
    assert store.load_checkpoint(command.run_id) == checkpoint
    assert store.load_session("session").context_window is None


def test_window_initialization_advances_revision_without_message_delta(
    store: LifecycleStore,
) -> None:
    command = replace(_input_command(store), message_delta=[])
    committed = store.commit_run_input(command)
    session = store.load_session("session")
    assert session.messages == []
    assert session.context_window == command.initial_context_window
    assert session.revision == committed.checkpoint.session_revision == 1
    assert committed.session_revision == 1


@pytest.mark.parametrize("empty_window", [False, True])
def test_input_reuses_initialized_window_across_runs(
    store: LifecycleStore, empty_window: bool
) -> None:
    initial = SessionContextWindow() if empty_window else _window()
    first = store.commit_run_input(replace(_input_command(store), initial_context_window=initial))
    store.finish_run(
        FinishRun(
            run_id=first.run.run_id,
            expected_run_revision=first.run.revision,
            activation_id=first.run.current_activation_id,
            stop_reason=RunStopReason.COMPLETED,
            now=NOW,
        )
    )
    second = _input_command(store, run_id="second-run")
    before = store.load_session("session")
    with pytest.raises(IrisRunStateError, match="context window"):
        store.commit_run_input(replace(second, initial_context_window=_window(2)))
    assert store.load_session("session") == before

    committed = store.commit_run_input(replace(second, initial_context_window=None))
    saved = store.load_session("session")
    assert saved.context_window == initial
    assert saved.revision == before.revision + 1
    assert committed.checkpoint.session_revision == saved.revision
    assert saved.messages == [*before.messages, *second.message_delta]


def test_compaction_replaces_window_and_fork_starts_uninitialized(store: LifecycleStore) -> None:
    command = _input_command(store)
    first = store.commit_run_input(command)
    ready = store.reserve_model_step(
        ReserveModelStep(
            run_id=first.run.run_id,
            expected_run_revision=first.run.revision,
            activation_id=command.activation_id,
            now=NOW,
        )
    )
    compact = replace(_compaction_command(ready), context_window=_window(2))
    committed = store.commit_compaction(compact)
    saved = store.load_session("session")
    assert saved.messages == command.message_delta
    assert saved.context_window == _window(2)
    assert saved.compaction == compact.compaction
    assert saved.revision == 2
    assert committed.checkpoint.session_revision == saved.revision
    assert committed.checkpoint.checkpoint_version == 3
    if isinstance(store, SQLiteStore):
        reopened = SQLiteStore(store.path)
        assert reopened.load_session("session") == saved
        assert reopened.load_checkpoint(command.run_id) == committed.checkpoint
    store.finish_run(
        FinishRun(
            run_id=committed.run.run_id,
            expected_run_revision=committed.run.revision,
            activation_id=command.activation_id,
            stop_reason=RunStopReason.COMPLETED,
            now=NOW,
        )
    )
    branch = store.fork_session(
        ForkSession(source_run_id=command.run_id, target_session_id="branch", now=NOW)
    )
    assert branch.context_window is None
    assert branch.messages == saved.messages
    assert branch.compaction == saved.compaction
    branch_input = _input_command(store, run_id="branch-run", session_id="branch")
    store.commit_run_input(replace(branch_input, initial_context_window=_window(3)))
    assert store.load_session("branch").context_window == _window(3)
    assert store.load_session("session").context_window == _window(2)


@pytest.mark.parametrize("failure", ["session_revision", "cancelled", "checkpoint"])
def test_compaction_failure_preserves_initialized_window(
    store: LifecycleStore, failure: str
) -> None:
    command = _input_command(store)
    first = store.commit_run_input(command)
    ready = store.reserve_model_step(
        ReserveModelStep(
            run_id=first.run.run_id,
            expected_run_revision=first.run.revision,
            activation_id=command.activation_id,
            now=NOW,
        )
    )
    if failure == "cancelled":
        ready = store.request_cancellation(
            RequestCancellation(
                run_id=ready.run.run_id,
                expected_run_revision=ready.run.revision,
                activation_id=command.activation_id,
                reason="cancel before window replacement",
                now=NOW,
            )
        )
    compact = replace(_compaction_command(ready), context_window=_window(2))
    if failure == "session_revision":
        compact = replace(compact, expected_session_revision=0)
    elif failure == "checkpoint":
        compact = replace(
            compact, checkpoint=compact.checkpoint.model_copy(update={"session_revision": 3})
        )
    before = store.load_session("session")
    events = store.list_events(command.run_id)
    with pytest.raises((IrisRunStateError, IrisRunConflictError)):
        store.commit_compaction(compact)
    assert store.load_session("session") == before
    assert store.load_run(command.run_id) == ready.run
    assert store.load_checkpoint(command.run_id) == ready.checkpoint
    assert store.list_events(command.run_id) == events


def test_checkpoint_v2_is_rejected_at_load_boundary(tmp_path: Path) -> None:
    path = tmp_path / "old-checkpoint.db"
    store = SQLiteStore(path)
    command = _input_command(store)
    with pytest.raises(ValidationError, match="checkpoint_version"):
        RunCheckpoint.model_validate({**command.checkpoint.model_dump(), "checkpoint_version": 2})
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE run_checkpoints SET checkpoint_version = 2")
    with pytest.raises(IrisRunPersistenceError):
        store.load_checkpoint(command.run_id)


def test_input_sql_failure_rolls_back_history_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollback.db"
    store = SQLiteStore(path)
    command = _input_command(store)
    before = store.load_run(command.run_id)
    checkpoint_before = store.load_checkpoint(command.run_id)
    original = sqlite_module._execute

    def fail_checkpoint(
        connection: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()
    ) -> sqlite3.Cursor:
        if "UPDATE run_checkpoints" in sql:
            raise sqlite3.OperationalError("input checkpoint failure")
        return original(connection, sql, params)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite_module, "_execute", fail_checkpoint)
        with pytest.raises(IrisRunPersistenceError):
            store.commit_run_input(command)

    reopened = SQLiteStore(path)
    assert reopened.load_run(command.run_id) == before
    assert reopened.load_checkpoint(command.run_id) == checkpoint_before
    assert reopened.load_session("session").messages == []
    assert reopened.load_session("session").context_window is None
