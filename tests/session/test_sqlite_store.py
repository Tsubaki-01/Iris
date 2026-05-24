from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import load_agent_config
from iris.exceptions import IrisExecutionError
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

    with pytest.raises(IrisExecutionError):
        store.save_messages("session-1", [{"bad": object()}])


def test_sqlite_session_store_wraps_directory_creation_errors(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(IrisExecutionError):
        SQLiteSessionStore(parent_file / "session.db")
