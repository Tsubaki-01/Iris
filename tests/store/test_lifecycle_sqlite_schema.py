"""Lifecycle SQLite v5 schema 与 session history 的持久化契约测试。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from iris.exceptions import IrisLifecycleSchemaError, IrisRunPersistenceError
from iris.lifecycle import AgentRunOptions, FinishRun, ForkSession, RunLimits, RunStopReason
from iris.message import Msg
from iris.store import SQLiteStore

from .test_lifecycle_store_contract import (
    _complete_history_turn,
    _create_command,
    _prepare_tool_batch,
)

_TABLES = {
    "subagent_run_links",
    "agent_runs",
    "lifecycle_schema",
    "run_activations",
    "run_checkpoints",
    "run_events",
    "run_interactions",
    "run_tool_calls",
    "session_messages",
    "session_run_lanes",
    "sessions",
}
_COLUMNS = {
    "subagent_run_links": ["parent_run_id", "parent_tool_call_id", "child_run_id"],
    "lifecycle_schema": ["component", "version"],
    "sessions": ["session_id", "revision", "message_count", "updated_at", "forked_from_run_id"],
    "session_messages": ["session_id", "ordinal", "message_json"],
    "agent_runs": [
        "run_id",
        "session_id",
        "agent_id",
        "phase",
        "stop_reason",
        "request_json",
        "options_json",
        "environment_fingerprint",
        "session_revision",
        "run_revision",
        "current_activation_id",
        "pending_interaction_id",
        "cancellation_requested_at",
        "cancellation_reason",
        "usage_json",
        "assistant_message_json",
        "error_json",
        "checkpoint_sequence",
        "last_event_sequence",
        "created_at",
        "started_at",
        "updated_at",
        "finished_at",
        "terminal_session_message_count",
    ],
    "session_run_lanes": ["session_id", "run_id", "revision", "acquired_at"],
    "run_activations": [
        "activation_id",
        "run_id",
        "ordinal",
        "kind",
        "status",
        "outcome",
        "started_at",
        "ended_at",
    ],
    "run_checkpoints": [
        "run_id",
        "sequence",
        "activation_id",
        "checkpoint_version",
        "cursor_json",
        "session_revision",
        "model_steps_reserved",
        "model_steps_committed",
        "environment_fingerprint",
        "resumability",
        "updated_at",
    ],
    "run_tool_calls": [
        "run_id",
        "tool_call_id",
        "step_index",
        "ordinal",
        "tool_name",
        "arguments_json",
        "fingerprint",
        "interaction_id",
        "phase",
        "claim_activation_id",
        "result_json",
        "version",
        "prepared_at",
        "updated_at",
        "claimed_at",
        "committed_at",
    ],
    "run_interactions": [
        "interaction_id",
        "run_id",
        "session_id",
        "step_index",
        "tool_call_id",
        "status",
        "request_json",
        "response_json",
        "version",
        "expires_at",
        "created_at",
        "resolved_at",
        "closed_at",
        "close_reason",
    ],
    "run_events": [
        "run_id",
        "sequence",
        "session_id",
        "kind",
        "occurred_at",
        "activation_id",
        "step_index",
        "correlation_id",
        "payload_json",
    ],
}
_NOW = "2026-01-02T03:04:00+00:00"


def _message_json(text: str = "hello") -> str:
    return json.dumps(Msg.user(text).model_dump(mode="json"), ensure_ascii=False)


def test_empty_database_creates_exact_v5_schema_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.db"
    path.touch()

    first = SQLiteStore(path)
    second = SQLiteStore(path)

    assert first.path == path
    assert second.load_session("missing").revision == 0
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }
        identity = connection.execute("SELECT component, version FROM lifecycle_schema").fetchall()
        columns = {
            table: [row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()]
            for table in _TABLES
        }
        triggers = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }
        message_fks = connection.execute("PRAGMA foreign_key_list(session_messages)").fetchall()
        session_fks = connection.execute("PRAGMA foreign_key_list(sessions)").fetchall()
        link_fks = connection.execute("PRAGMA foreign_key_list(subagent_run_links)").fetchall()
        terminal_index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'terminal_runs_by_session'"
        ).fetchone()

    assert tables == _TABLES
    assert indexes == {"one_open_interaction_per_run", "terminal_runs_by_session"}
    assert triggers == set()
    assert identity == [("agent_lifecycle", 5)]
    assert columns == _COLUMNS
    assert [(row[2], row[3], row[4]) for row in message_fks] == [
        ("sessions", "session_id", "session_id")
    ]
    assert [(row[2], row[3], row[4]) for row in session_fks] == [
        ("agent_runs", "forked_from_run_id", "run_id")
    ]
    assert " ".join(terminal_index[0].split()) == (
        "CREATE INDEX terminal_runs_by_session ON agent_runs(session_id, created_at, run_id) "
        "WHERE phase = 'terminal'"
    )
    assert {(row[2], row[3], row[4]) for row in link_fks} == {
        ("run_tool_calls", "parent_run_id", "run_id"),
        ("run_tool_calls", "parent_tool_call_id", "tool_call_id"),
        ("agent_runs", "child_run_id", "run_id"),
    }


@pytest.mark.parametrize("kind", ["legacy", "v3", "v4", "extra", "missing", "unknown_version"])
def test_incompatible_database_is_rejected_without_changing_bytes(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / f"{kind}.db"
    if kind == "legacy":
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE lifecycle_schema "
                "(component TEXT PRIMARY KEY, version INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO lifecycle_schema(component, version) VALUES ('agent_lifecycle', 1)"
            )
            connection.execute(
                "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, messages_json TEXT NOT NULL)"
            )
    else:
        SQLiteStore(path)
        with sqlite3.connect(path) as connection:
            if kind == "extra":
                connection.execute("CREATE TABLE unexpected (value TEXT)")
            elif kind == "missing":
                connection.execute("DROP TABLE run_events")
            elif kind == "v3":
                connection.execute("DROP TABLE subagent_run_links")
                connection.execute("UPDATE lifecycle_schema SET version = 3")
            elif kind == "v4":
                connection.execute("UPDATE lifecycle_schema SET version = 4")
            else:
                connection.execute("UPDATE lifecycle_schema SET version = 99")
    before = path.read_bytes()

    with pytest.raises(IrisLifecycleSchemaError):
        SQLiteStore(path)

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("message_count", "rows"),
    [
        (1, [(1, "not-json")]),
        (1, [(1, "1")]),
        (2, [(1, _message_json("one"))]),
        (2, [(1, _message_json("one")), (3, _message_json("three"))]),
        (1, [(2, _message_json("two"))]),
        (1, [(1, _message_json("one")), (2, _message_json("extra"))]),
    ],
    ids=["invalid-json", "invalid-message", "count-mismatch", "gap", "not-one-based", "extra-row"],
)
def test_corrupt_session_messages_are_mapped_to_persistence_error(
    tmp_path: Path,
    message_count: int,
    rows: list[tuple[int, str]],
) -> None:
    path = tmp_path / "lifecycle.db"
    store = SQLiteStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO sessions(session_id, revision, message_count, updated_at)
            VALUES ('broken', 1, ?, ?)""",
            (message_count, _NOW),
        )
        connection.executemany(
            """INSERT INTO session_messages(session_id, ordinal, message_json)
            VALUES ('broken', ?, ?)""",
            rows,
        )

    with pytest.raises(IrisRunPersistenceError) as captured:
        store.load_session("broken")

    assert captured.value.context["operation"] == "load_session"
    assert captured.value.context["path"] == str(path)


@pytest.mark.parametrize("ignore_checks", [False, True], ids=["sql-check", "row-load"])
@pytest.mark.parametrize(
    ("phase", "count"),
    [("active", 0), ("terminal", None), ("terminal", -1)],
)
def test_terminal_cutoff_constraints_apply_to_sql_and_loaded_rows(
    tmp_path: Path,
    phase: str,
    count: int | None,
    ignore_checks: bool,
) -> None:
    """SQL 约束和原始行解析都拒绝与运行阶段不符的消息截点。"""
    store = SQLiteStore(tmp_path / "cutoff.db")
    command = _create_command()
    created = store.create_run(command)
    if phase == "terminal":
        store.finish_run(
            FinishRun(
                run_id="run-1",
                expected_run_revision=created.run.revision,
                activation_id="activation-1",
                stop_reason=RunStopReason.COMPLETED,
                now=command.now,
            )
        )
    with sqlite3.connect(store.path) as connection:
        statement = (
            "UPDATE agent_runs SET terminal_session_message_count = ? WHERE run_id = 'run-1'"
        )
        if ignore_checks:
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(statement, (count,))
        else:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement, (count,))
            return

    reopened = SQLiteStore(store.path)
    with pytest.raises(IrisRunPersistenceError) as captured:
        reopened.load_run("run-1")
    assert captured.value.context["operation"] == "load_run"


def test_session_lineage_survives_message_append_and_reopen(tmp_path: Path) -> None:
    """追加消息只推进会话版本和消息数，重开后直接来源仍可读取。"""
    store = SQLiteStore(tmp_path / "lineage.db")
    source_command = _create_command(
        run_id="source-run", session_id="source-session", activation_id="source-activation"
    )
    store.create_run(
        replace(
            source_command,
            options=AgentRunOptions(limits=RunLimits(deadline_at=source_command.now)),
        )
    )
    store.fork_session(
        ForkSession(
            source_run_id="source-run", target_session_id="session-1", now=source_command.now
        )
    )

    _prepare_tool_batch(store)
    for current_store in (store, SQLiteStore(store.path)):
        session = current_store.load_session("session-1")
        assert session.forked_from_run_id == "source-run"
        assert session.revision == 1
        assert len(session.messages) == 1
        assert current_store.load_run("source-run").terminal_session_message_count == 0


def test_reopened_store_lists_previews_and_forks_without_execution_facts(tmp_path: Path) -> None:
    """重开后可读取旧截点并 fork，新增资源仅为 session 与消息行。"""
    path = tmp_path / "history.db"
    store = SQLiteStore(path)
    first = _complete_history_turn(store, run_id="r1", session_id="main")
    expected = store.load_session("main")
    _complete_history_turn(store, run_id="r2", session_id="main")
    fact_tables = sorted(_TABLES - {"sessions", "session_messages", "lifecycle_schema"})
    with sqlite3.connect(path) as connection:
        counts_before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in fact_tables
        }
    reopened = SQLiteStore(path)
    assert [point.message_count for point in reopened.list_fork_points("main").items] == [2, 4]
    assert reopened.load_session_at_run("r1").messages == tuple(expected.messages)
    branch = reopened.fork_session(
        ForkSession(source_run_id="r1", target_session_id="branch", now=first.finished_at)
    )
    assert branch.messages == expected.messages
    assert branch.revision == 0
    assert branch.forked_from_run_id == "r1"
    assert SQLiteStore(path).load_session("branch") == branch
    with sqlite3.connect(path) as connection:
        counts_after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in fact_tables
        }
        source_rows = connection.execute(
            "SELECT ordinal, message_json FROM session_messages "
            "WHERE session_id = 'main' AND ordinal <= 2 ORDER BY ordinal"
        ).fetchall()
        target_rows = connection.execute(
            "SELECT ordinal, message_json FROM session_messages "
            "WHERE session_id = 'branch' ORDER BY ordinal"
        ).fetchall()
    assert counts_after == counts_before
    assert target_rows == source_rows


def test_preview_and_fork_only_decode_selected_prefix(tmp_path: Path) -> None:
    """旧截点读取不触及后续消息；完整历史读取仍识别后续损坏。"""
    store = SQLiteStore(tmp_path / "prefix.db")
    first = _complete_history_turn(store, run_id="r1", session_id="main")
    expected = store.load_session("main").messages
    _complete_history_turn(store, run_id="r2", session_id="main")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE session_messages SET message_json = 'not-json' "
            "WHERE session_id = 'main' AND ordinal = 3"
        )
    assert [point.run_id for point in store.list_fork_points("main").items] == ["r1", "r2"]
    assert store.load_session_at_run("r1").messages == tuple(expected)
    assert (
        store.fork_session(
            ForkSession(source_run_id="r1", target_session_id="branch", now=first.finished_at)
        ).messages
        == expected
    )
    with pytest.raises(IrisRunPersistenceError):
        store.load_session("main")
    with pytest.raises(IrisRunPersistenceError):
        store.load_session_at_run("r2")
