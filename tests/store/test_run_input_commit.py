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
    LifecycleStore,
    RunCheckpoint,
)
from iris.message import Msg
from iris.store import InMemoryLifecycleStore, SQLiteStore

NOW = datetime(2026, 9, 19, tzinfo=UTC)


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """同一契约覆盖两种 store。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "input.db")
    return InMemoryLifecycleStore()


def _input_command(store: LifecycleStore) -> CommitRunInput:
    checkpoint = RunCheckpoint(
        run_id="run-input",
        sequence=1,
        activation_id="activation-input",
        engine_cursor={"position": "before_input", "step_index": 0},
        session_revision=0,
        model_steps_reserved=0,
        model_steps_committed=0,
    )
    created = store.create_run(
        CreateRun(
            request=AgentRunRequest(input="question", session_id="session", run_id="run-input"),
            options=AgentRunOptions(),
            agent_id="agent",
            start_activation_id="activation-input",
            initial_checkpoint=checkpoint,
            now=NOW,
        )
    )
    return CommitRunInput(
        run_id=created.run.run_id,
        expected_run_revision=created.run.revision,
        activation_id="activation-input",
        expected_session_revision=0,
        message_delta=[Msg.user("memory snapshot"), Msg.user("BCI"), Msg.user("question")],
        checkpoint=checkpoint.model_copy(
            update={
                "sequence": 2,
                "session_revision": 1,
                "engine_cursor": {"position": "before_model", "step_index": 0},
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
    assert committed.checkpoint.checkpoint_version == 2
    assert committed.run.checkpoint_sequence == 2
    assert store.load_checkpoint(command.run_id) == command.checkpoint
    assert store.load_session("session").messages == command.message_delta
    assert store.list_events(command.run_id) == events
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
                }
            }
        command = replace(command, checkpoint=command.checkpoint.model_copy(update=changes))

    with pytest.raises((IrisRunConflictError, IrisRunStateError)):
        store.commit_run_input(command)

    assert store.load_run(command.run_id) == before
    assert store.load_checkpoint(command.run_id) == checkpoint_before
    assert store.load_session("session").messages == []


def test_checkpoint_v1_is_rejected_at_load_boundary(tmp_path: Path) -> None:
    path = tmp_path / "old-checkpoint.db"
    store = SQLiteStore(path)
    command = _input_command(store)
    with pytest.raises(ValidationError, match="checkpoint_version"):
        RunCheckpoint.model_validate(
            {**command.checkpoint.model_dump(), "checkpoint_version": 1}
        )
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE run_checkpoints SET checkpoint_version = 1")
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
