"""Lifecycle SQLite v4 schema 与 session history 的硬边界测试。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisLifecycleSchemaError, IrisRunPersistenceError
from iris.message import Msg
from iris.store import SQLiteStore

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
    "sessions": ["session_id", "revision", "message_count", "updated_at"],
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


def test_empty_database_creates_exact_v4_schema_and_reopens(tmp_path: Path) -> None:
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
        link_fks = connection.execute("PRAGMA foreign_key_list(subagent_run_links)").fetchall()

    assert tables == _TABLES
    assert indexes == {"one_open_interaction_per_run"}
    assert triggers == set()
    assert identity == [("agent_lifecycle", 4)]
    assert columns == _COLUMNS
    assert [(row[2], row[3], row[4]) for row in message_fks] == [
        ("sessions", "session_id", "session_id")
    ]
    assert {(row[2], row[3], row[4]) for row in link_fks} == {
        ("run_tool_calls", "parent_run_id", "run_id"),
        ("run_tool_calls", "parent_tool_call_id", "tool_call_id"),
        ("agent_runs", "child_run_id", "run_id"),
    }


@pytest.mark.parametrize("kind", ["legacy", "v3", "extra", "missing", "unknown_version"])
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
    ],
    ids=["invalid-json", "invalid-message", "count-mismatch", "gap", "not-one-based"],
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
