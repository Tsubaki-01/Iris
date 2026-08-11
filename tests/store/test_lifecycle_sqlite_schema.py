"""Lifecycle SQLite v1 schema 的硬边界测试。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisLifecycleSchemaError, IrisRunPersistenceError
from iris.store import SQLiteStore

_TABLES = {
    "agent_runs",
    "lifecycle_schema",
    "run_activations",
    "run_checkpoints",
    "run_events",
    "run_interactions",
    "run_tool_calls",
    "session_run_lanes",
    "sessions",
}


def test_empty_database_creates_exact_v1_schema_and_reopens(tmp_path: Path) -> None:
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
        interaction_fks = connection.execute("PRAGMA foreign_key_list(run_interactions)").fetchall()

    assert tables == _TABLES
    assert indexes == {"one_open_interaction_per_run"}
    assert identity == [("agent_lifecycle", 1)]
    assert [(row[2], row[3], row[4]) for row in interaction_fks] == [
        ("agent_runs", "run_id", "run_id")
    ]


@pytest.mark.parametrize("kind", ["old", "extra", "missing", "unknown_version"])
def test_incompatible_database_is_rejected_without_changing_bytes(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / f"{kind}.db"
    if kind == "old":
        with sqlite3.connect(path) as connection:
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


def test_corrupt_session_message_is_mapped_to_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.db"
    store = SQLiteStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO sessions(session_id, revision, messages_json, updated_at)
            VALUES ('broken', 0, '[1]', '2026-01-02T03:04:00+00:00')"""
        )

    with pytest.raises(IrisRunPersistenceError) as captured:
        store.load_session("broken")

    assert captured.value.context["operation"] == "load_session"
    assert captured.value.context["path"] == str(path)
