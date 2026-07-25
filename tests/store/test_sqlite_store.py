from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisSessionError
from iris.store import SQLiteStore

_V1_COLUMNS = [
    "interaction_id TEXT PRIMARY KEY",
    "session_id TEXT NOT NULL",
    "run_id TEXT NOT NULL",
    "step_index INTEGER NOT NULL",
    "tool_call_id TEXT NOT NULL",
    "kind TEXT NOT NULL",
    "status TEXT NOT NULL",
    "resume_phase TEXT NOT NULL",
    "request_json TEXT NOT NULL",
    "response_json TEXT",
    "checkpoint_json TEXT NOT NULL",
    "version INTEGER NOT NULL DEFAULT 1",
    "created_at TEXT NOT NULL",
    "resolved_at TEXT",
    "consumed_at TEXT",
]

_V2_COLUMNS = [
    "interaction_id TEXT PRIMARY KEY",
    "session_id TEXT NOT NULL",
    "run_id TEXT NOT NULL",
    "step_index INTEGER NOT NULL",
    "status TEXT NOT NULL",
    "resume_phase TEXT NOT NULL",
    "request_json TEXT NOT NULL",
    "response_json TEXT",
    "checkpoint_json TEXT NOT NULL",
    "version INTEGER NOT NULL DEFAULT 1",
    "created_at TEXT NOT NULL",
    "resolved_at TEXT",
    "consumed_at TEXT",
]

_V1_SIGNATURE = (
    ("interaction_id", "TEXT", 0, None, 1),
    ("session_id", "TEXT", 1, None, 0),
    ("run_id", "TEXT", 1, None, 0),
    ("step_index", "INTEGER", 1, None, 0),
    ("tool_call_id", "TEXT", 1, None, 0),
    ("kind", "TEXT", 1, None, 0),
    ("status", "TEXT", 1, None, 0),
    ("resume_phase", "TEXT", 1, None, 0),
    ("request_json", "TEXT", 1, None, 0),
    ("response_json", "TEXT", 0, None, 0),
    ("checkpoint_json", "TEXT", 1, None, 0),
    ("version", "INTEGER", 1, "1", 0),
    ("created_at", "TEXT", 1, None, 0),
    ("resolved_at", "TEXT", 0, None, 0),
    ("consumed_at", "TEXT", 0, None, 0),
)

_V2_SIGNATURE = tuple(
    column for column in _V1_SIGNATURE if column[0] not in {"tool_call_id", "kind"}
)


def test_sqlite_store_persists_messages(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "session.db")
    messages = [{"role": "user", "content": "你好"}]

    store.save_messages("session-1", messages)

    assert store.load_messages("session-1") == messages
    assert (tmp_path / "session.db").is_file()


def test_sqlite_store_persists_run_metadata(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "session.db")
    metadata = {"model": "openai/gpt-4o-mini", "status": "ok"}

    store.save_run_metadata("session-1", metadata)

    assert store.load_run_metadata("session-1") == metadata


def test_sqlite_store_appends_tool_events(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "session.db")

    store.append_tool_event(
        "session-1",
        {
            "event_id": "tool_result:run_1:call_1",
            "tool_name": "read_file",
            "status": "ok",
        },
    )
    store.append_tool_event(
        "session-1",
        {
            "event_id": "tool_result:run_1:call_2",
            "tool_name": "grep_search",
            "status": "error",
        },
    )

    assert store.load_tool_events("session-1") == [
        {
            "tool_name": "read_file",
            "status": "ok",
            "event_id": "tool_result:run_1:call_1",
        },
        {
            "tool_name": "grep_search",
            "status": "error",
            "event_id": "tool_result:run_1:call_2",
        },
    ]


def test_sqlite_store_appends_same_event_once(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "session.db")
    event_id = "tool_result:run_1:call_1"

    store.append_tool_event("session-1", {"event_id": event_id, "status": "ok", "count": 1})
    store.append_tool_event("session-1", {"count": 1, "status": "ok", "event_id": event_id})

    assert store.load_tool_events("session-1") == [
        {"status": "ok", "count": 1, "event_id": event_id}
    ]


def test_sqlite_store_rejects_different_idempotent_event_payload(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "session.db")
    store.append_tool_event(
        "session-1",
        {"event_id": "tool_result:run_1:call_1", "status": "ok"},
    )

    with pytest.raises(IrisSessionError):
        store.append_tool_event(
            "session-1",
            {"event_id": "tool_result:run_1:call_1", "status": "error"},
        )


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"event_id": "  "},
        {"event_id": 1},
        {"event_id": "tool_result:run_1:call_1", "not_json": {"set"}},
    ],
)
def test_sqlite_store_rejects_invalid_idempotent_event(
    tmp_path: Path,
    event: dict[str, object],
) -> None:
    store = SQLiteStore(tmp_path / "session.db")

    with pytest.raises(IrisSessionError):
        store.append_tool_event("session-1", event)


def test_sqlite_store_rejects_inconsistent_existing_events(tmp_path: Path) -> None:
    path = tmp_path / "session.db"
    store = SQLiteStore(path)
    store.save_messages("session-1", [])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE sessions SET tool_events_json = ? WHERE session_id = ?",
            (
                json.dumps(
                    [
                        {"event_id": "tool_result:run_1:call_1", "status": "ok"},
                        {"event_id": "tool_result:run_1:call_1", "status": "error"},
                    ]
                ),
                "session-1",
            ),
        )

    with pytest.raises(IrisSessionError):
        store.append_tool_event(
            "session-1",
            {"event_id": "tool_result:run_1:call_1", "status": "ok"},
        )


def test_sqlite_store_returns_empty_defaults_for_missing_session(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "session.db")

    assert store.load_messages("missing") == []
    assert store.load_run_metadata("missing") == {}
    assert store.load_tool_events("missing") == []


def test_backend_none_config_does_not_create_database(tmp_path: Path) -> None:
    from iris.agents import load_agent_config

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        """
name: no-session
model: openai/gpt-4o-mini
system: 你是一个本地助手。
session:
  backend: none
  path: .iris/session.db
""",
        encoding="utf-8",
    )

    config = load_agent_config(config_path)

    assert config.session.backend == "none"
    assert not (tmp_path / ".iris" / "session.db").exists()


def test_sqlite_store_rejects_non_json_values(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "session.db")

    with pytest.raises(IrisSessionError):
        store.save_messages("session-1", [{"bad": object()}])


def test_sqlite_store_wraps_directory_creation_errors(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(IrisSessionError):
        SQLiteStore(parent_file / "session.db")


def test_sqlite_store_auto_upgrades_sessions_only_database(tmp_path: Path) -> None:
    path = tmp_path / "session.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL DEFAULT '[]',
                run_metadata_json TEXT NOT NULL DEFAULT '{}',
                tool_events_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            )
            """
        )

    SQLiteStore(path)

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        indexes = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    assert "human_interactions" in tables
    assert "idx_human_interactions_active_session" in indexes
    assert _hitl_signature(path) == _V2_SIGNATURE


def test_sqlite_store_keeps_exact_v2_schema_and_restores_indexes(tmp_path: Path) -> None:
    path = tmp_path / "session.db"
    with sqlite3.connect(path) as connection:
        _create_sessions_table(connection)
        _create_hitl_table(connection, _V2_COLUMNS)
        connection.execute(
            """
            INSERT INTO human_interactions (
                interaction_id, session_id, run_id, step_index, status, resume_phase,
                request_json, checkpoint_json, version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "int_keep",
                "session-1",
                "run-1",
                0,
                "pending",
                "waiting",
                "{}",
                "{}",
                1,
                "2026-07-23T00:00:00",
            ),
        )

    SQLiteStore(path)
    first_signature = _hitl_signature(path)
    SQLiteStore(path)

    with sqlite3.connect(path) as connection:
        indexes = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        rows = connection.execute(
            "SELECT interaction_id, session_id, run_id FROM human_interactions"
        ).fetchall()
    assert first_signature == _V2_SIGNATURE
    assert _hitl_signature(path) == _V2_SIGNATURE
    assert "idx_human_interactions_session_status_phase" in indexes
    assert "idx_human_interactions_active_session" in indexes
    assert rows == [("int_keep", "session-1", "run-1")]


def test_sqlite_store_rejects_exact_v1_without_modifying_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.db"
    metadata = {
        "latest_run": {
            "status": "waiting_human",
            "waiting_human": True,
            "interaction_id": "int_old",
            "keep": "value",
        },
        "other": "metadata",
    }
    with sqlite3.connect(path) as connection:
        _create_sessions_table(connection)
        connection.execute(
            """
            INSERT INTO sessions (
                session_id, messages_json, run_metadata_json, tool_events_json, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "session-1",
                json.dumps([{"role": "user", "content": "keep"}]),
                json.dumps(metadata),
                json.dumps([{"event_id": "event-1", "status": "ok"}]),
                "2026-07-23T00:00:00",
            ),
        )
        untouched_metadata = {
            "latest_run": {"status": "ok", "keep": "untouched"},
            "other": "untouched",
        }
        connection.execute(
            """
            INSERT INTO sessions (
                session_id, messages_json, run_metadata_json, tool_events_json, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "session-2",
                json.dumps([{"role": "user", "content": "untouched"}]),
                json.dumps(untouched_metadata),
                json.dumps([{"event_id": "event-2", "status": "ok"}]),
                "2026-07-23T00:00:00",
            ),
        )
        _create_hitl_table(connection, _V1_COLUMNS)
        connection.execute(
            """
            INSERT INTO human_interactions (
                interaction_id, session_id, run_id, step_index, tool_call_id, kind,
                status, resume_phase, request_json, response_json, checkpoint_json,
                version, created_at, resolved_at, consumed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "int_old",
                "session-1",
                "run-1",
                0,
                "call-1",
                "permission",
                "pending",
                "waiting",
                "{}",
                None,
                "{}",
                1,
                "2026-07-23T00:00:00",
                None,
                None,
            ),
        )

    with sqlite3.connect(path) as connection:
        original_sessions = connection.execute(
            "SELECT * FROM sessions ORDER BY session_id"
        ).fetchall()
        original_interactions = connection.execute("SELECT * FROM human_interactions").fetchall()
        original_indexes = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' ORDER BY name"
        ).fetchall()

    with pytest.raises(IrisSessionError, match="v1"):
        SQLiteStore(path)

    assert _hitl_signature(path) == _V1_SIGNATURE
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM sessions ORDER BY session_id").fetchall() == (
            original_sessions
        )
        assert connection.execute("SELECT * FROM human_interactions").fetchall() == (
            original_interactions
        )
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' ORDER BY name"
            ).fetchall()
            == original_indexes
        )


@pytest.mark.parametrize(
    "columns",
    [
        _V2_COLUMNS[:-1],
        [*_V2_COLUMNS, "unexpected TEXT"],
        [_V2_COLUMNS[0], _V2_COLUMNS[2], _V2_COLUMNS[1], *_V2_COLUMNS[3:]],
        [column.replace("run_id TEXT", "run_id BLOB") for column in _V2_COLUMNS],
        [column.replace("session_id TEXT NOT NULL", "session_id TEXT") for column in _V2_COLUMNS],
        [column.replace("DEFAULT 1", "DEFAULT 2") for column in _V2_COLUMNS],
        [
            column.replace("interaction_id TEXT PRIMARY KEY", "interaction_id TEXT")
            for column in _V2_COLUMNS
        ],
    ],
    ids=[
        "missing-column",
        "extra-column",
        "reordered-columns",
        "type",
        "nullability",
        "default",
        "primary-key",
    ],
)
def test_sqlite_store_rejects_unknown_hitl_schema_without_dropping_it(
    tmp_path: Path,
    columns: list[str],
) -> None:
    path = tmp_path / "session.db"
    with sqlite3.connect(path) as connection:
        _create_sessions_table(connection)
        _create_hitl_table(connection, columns)
        connection.execute(
            "INSERT INTO sessions (session_id, updated_at) VALUES ('session-1', 'before')"
        )
    original_signature = _hitl_signature(path)
    with sqlite3.connect(path) as connection:
        original_sessions = connection.execute("SELECT * FROM sessions").fetchall()
        original_interactions = connection.execute("SELECT * FROM human_interactions").fetchall()

    with pytest.raises(IrisSessionError, match="schema"):
        SQLiteStore(path)

    assert _hitl_signature(path) == original_signature
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM sessions").fetchall() == original_sessions
        assert connection.execute("SELECT * FROM human_interactions").fetchall() == (
            original_interactions
        )


def _create_sessions_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            messages_json TEXT NOT NULL DEFAULT '[]',
            run_metadata_json TEXT NOT NULL DEFAULT '{}',
            tool_events_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        )
        """
    )


def _create_hitl_table(connection: sqlite3.Connection, columns: list[str]) -> None:
    connection.execute(f"CREATE TABLE human_interactions ({', '.join(columns)})")


def _hitl_signature(path: Path) -> tuple[tuple[object, ...], ...]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("PRAGMA table_info(human_interactions)").fetchall()
    return tuple((row[1], str(row[2]).upper(), row[3], row[4], row[5]) for row in rows)
