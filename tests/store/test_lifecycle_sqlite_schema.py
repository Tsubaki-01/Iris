"""Lifecycle SQLite v2 schema 与 session history 的硬边界测试。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisLifecycleSchemaError, IrisRunPersistenceError
from iris.message import Msg
from iris.store import SQLiteStore

_TABLES = {
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
_NOW = "2026-01-02T03:04:00+00:00"


def _message_json(text: str = "hello") -> str:
    return json.dumps(Msg.user(text).model_dump(mode="json"), ensure_ascii=False)


def test_empty_database_creates_exact_v2_schema_and_reopens(tmp_path: Path) -> None:
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
        session_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        ]
        message_fks = connection.execute("PRAGMA foreign_key_list(session_messages)").fetchall()

    assert tables == _TABLES
    assert indexes == {"one_open_interaction_per_run"}
    assert identity == [("agent_lifecycle", 2)]
    assert session_columns == ["session_id", "revision", "message_count", "updated_at"]
    assert [(row[2], row[3], row[4]) for row in message_fks] == [
        ("sessions", "session_id", "session_id")
    ]


@pytest.mark.parametrize("kind", ["legacy", "extra", "missing", "unknown_version"])
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
