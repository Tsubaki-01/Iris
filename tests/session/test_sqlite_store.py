from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris.agents import load_agent_config
from iris.exceptions import IrisSessionError
from iris.session import SQLiteSessionStore


def test_sqlite_session_store_persists_messages(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")
    messages = [{"role": "user", "content": "你好"}]

    store.save_messages("session-1", messages)

    assert store.load_messages("session-1") == messages
    assert (tmp_path / "session.db").is_file()


def test_sqlite_session_store_persists_run_metadata(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")
    metadata = {"model": "openai/gpt-4o-mini", "status": "ok"}

    store.save_run_metadata("session-1", metadata)

    assert store.load_run_metadata("session-1") == metadata


def test_sqlite_session_store_appends_tool_events(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")

    store.append_tool_event("session-1", {"tool_name": "read_file", "status": "ok"})
    store.append_tool_event("session-1", {"tool_name": "grep_search", "status": "error"})

    assert store.load_tool_events("session-1") == [
        {"tool_name": "read_file", "status": "ok"},
        {"tool_name": "grep_search", "status": "error"},
    ]


def test_sqlite_session_store_appends_same_idempotent_event_once(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")
    event_id = "tool_result:run_1:call_1"

    store.append_tool_event_once("session-1", event_id, {"status": "ok", "count": 1})
    store.append_tool_event_once("session-1", event_id, {"count": 1, "status": "ok"})

    assert store.load_tool_events("session-1") == [
        {"status": "ok", "count": 1, "event_id": event_id}
    ]


def test_sqlite_session_store_rejects_different_idempotent_event_payload(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")
    store.append_tool_event_once("session-1", "tool_result:run_1:call_1", {"status": "ok"})

    with pytest.raises(IrisSessionError):
        store.append_tool_event_once("session-1", "tool_result:run_1:call_1", {"status": "error"})


def test_sqlite_session_store_rejects_non_json_idempotent_event(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")

    with pytest.raises(IrisSessionError):
        store.append_tool_event_once("session-1", "tool_result:run_1:call_1", {"not_json": {"set"}})


def test_sqlite_session_store_returns_empty_defaults_for_missing_session(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")

    assert store.load_messages("missing") == []
    assert store.load_run_metadata("missing") == {}
    assert store.load_tool_events("missing") == []


def test_backend_none_config_does_not_create_database(tmp_path: Path) -> None:
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


def test_sqlite_session_store_rejects_non_json_values(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "session.db")

    with pytest.raises(IrisSessionError):
        store.save_messages("session-1", [{"bad": object()}])


def test_sqlite_session_store_wraps_directory_creation_errors(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(IrisSessionError):
        SQLiteSessionStore(parent_file / "session.db")


def test_sqlite_session_store_auto_upgrades_sessions_only_database(tmp_path: Path) -> None:
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

    SQLiteSessionStore(path)

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
