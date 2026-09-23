"""记忆取材读取本 run 新增消息，并保持来源身份与单次快照一致。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from iris.exceptions import IrisRunNotFoundError, IrisRunStateError
from iris.lifecycle import FinishRun, ForkSession, LifecycleStore, RunRecord, RunStopReason
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .test_lifecycle_store_contract import _NOW, _complete_history_turn
from .test_run_input_commit import _input_command


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """两个实现共享来源切片契约。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "source.db")
    return InMemoryLifecycleStore()


def test_source_identity_is_stable_within_store_lifetime(store: LifecycleStore) -> None:
    identity = store.source_id
    assert str(UUID(identity)) == identity
    _input_command(store)
    assert store.source_id == identity
    if isinstance(store, SQLiteStore):
        assert SQLiteStore(store.path).source_id == identity
    else:
        assert InMemoryLifecycleStore().source_id != identity


def test_distinct_sqlite_databases_have_distinct_sources(tmp_path: Path) -> None:
    assert (
        SQLiteStore(tmp_path / "first.db").source_id
        != SQLiteStore(tmp_path / "second.db").source_id
    )


def test_active_slice_reads_only_committed_suffix(store: LifecycleStore) -> None:
    command = _input_command(store)
    before = store.load_run_message_slice(command.run_id)
    assert before.source_id == store.source_id
    assert before.run_id == command.run_id
    assert before.session_id == "session"
    assert (
        before.initial_message_count == before.start_message_count == before.end_message_count == 0
    )
    assert before.messages == ()
    assert before.terminal_message_count is None
    assert before.outcome is None

    store.commit_run_input(command)
    committed = store.load_run_message_slice(command.run_id, 1)
    assert committed.initial_message_count == 0
    assert committed.start_message_count == 1
    assert committed.end_message_count == 2
    assert committed.messages == tuple(command.message_delta[1:])
    assert committed.terminal_message_count is None
    assert committed.outcome is None
    committed.messages[0].metadata["changed"] = True
    assert store.load_run_message_slice(command.run_id, 1).messages == tuple(
        command.message_delta[1:]
    )
    assert store.load_run_message_slice(command.run_id, 2).messages == ()


def test_terminal_slice_stops_before_later_run_and_skips_previous_run(
    store: LifecycleStore,
) -> None:
    _complete_history_turn(store, run_id="first", session_id="main")
    second = _complete_history_turn(store, run_id="second", session_id="main")
    expected = tuple(store.load_session("main").messages[2:])
    _complete_history_turn(store, run_id="third", session_id="main")

    sliced = store.load_run_message_slice("second")
    assert sliced.initial_message_count == sliced.start_message_count == 2
    assert sliced.end_message_count == sliced.terminal_message_count == 4
    assert sliced.outcome is RunStopReason.COMPLETED
    assert sliced.messages == expected
    assert sliced.terminal_message_count == second.terminal_session_message_count
    assert store.load_run_message_slice("second", 3).messages == expected[1:]
    assert store.load_run_message_slice("second", 4).messages == ()


def test_fork_inherited_history_is_not_new_source_material(store: LifecycleStore) -> None:
    _complete_history_turn(store, run_id="source", session_id="main")
    store.fork_session(ForkSession(source_run_id="source", target_session_id="branch", now=_NOW))
    command = _input_command(store, run_id="branch-run", session_id="branch")
    before = store.load_run_message_slice("branch-run")
    assert (
        before.initial_message_count == before.start_message_count == before.end_message_count == 2
    )
    assert before.messages == ()

    store.commit_run_input(command)
    sliced = store.load_run_message_slice("branch-run")
    assert sliced.initial_message_count == sliced.start_message_count == 2
    assert sliced.end_message_count == 4
    assert sliced.messages == tuple(command.message_delta)


def test_source_slice_rejects_invalid_cursor_and_unknown_run(store: LifecycleStore) -> None:
    command = _input_command(store)
    store.commit_run_input(command)
    with pytest.raises(IrisRunNotFoundError):
        store.load_run_message_slice("missing")
    for after_count in (-1, 3):
        with pytest.raises(IrisRunStateError):
            store.load_run_message_slice(command.run_id, after_count)


def test_empty_terminal_run_preserves_zero_cutoff_and_outcome(store: LifecycleStore) -> None:
    command = _input_command(store)
    store.finish_run(
        FinishRun(
            run_id=command.run_id,
            expected_run_revision=command.expected_run_revision,
            activation_id=command.activation_id,
            stop_reason=RunStopReason.CANCELLED,
            now=command.now,
        )
    )
    sliced = store.load_run_message_slice(command.run_id)
    assert sliced.initial_message_count == sliced.start_message_count == 0
    assert sliced.end_message_count == sliced.terminal_message_count == 0
    assert sliced.outcome is RunStopReason.CANCELLED
    assert sliced.messages == ()


def test_sqlite_slice_does_not_decode_messages_outside_requested_interval(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "bounded.db")
    _complete_history_turn(store, run_id="first", session_id="main")
    _complete_history_turn(store, run_id="second", session_id="main")
    expected = tuple(store.load_session("main").messages[2:])
    _complete_history_turn(store, run_id="third", session_id="main")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """UPDATE session_messages SET message_json = 'invalid'
            WHERE session_id = 'main' AND ordinal IN (1, 6)"""
        )

    assert store.load_run_message_slice("second").messages == expected


def test_sqlite_slice_metadata_and_messages_share_read_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteStore(tmp_path / "concurrent.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
    writer = SQLiteStore(store.path)
    command = _input_command(store)
    original = store._select_run
    advanced = False

    def read_run(
        connection: sqlite3.Connection, run_id: str, *, operation: str
    ) -> RunRecord | None:
        nonlocal advanced
        run = original(connection, run_id, operation=operation)
        if not advanced:
            advanced = True
            writer.commit_run_input(command)
        return run

    monkeypatch.setattr(store, "_select_run", read_run)
    first = store.load_run_message_slice(command.run_id)
    assert first.end_message_count == 0
    assert first.messages == ()
    second = store.load_run_message_slice(command.run_id)
    assert second.end_message_count == 2
    assert second.messages == tuple(command.message_delta)
