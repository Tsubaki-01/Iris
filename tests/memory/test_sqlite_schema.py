"""Memory schema v2 初始化与 FTS 错误边界。"""

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory.models import MemoryQuery
from iris.memory.sqlite import SQLiteMemoryStore


def test_schema_v2_stores_only_namespace_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    SQLiteMemoryStore(path)
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM memory_schema WHERE key='schema_version'"
        ).fetchone()
        assert version == ("2",)
        for table in ["memory_items", "memory_candidates", "memory_episodes", "memory_events"]:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            assert "namespace" in columns
            assert not any(column.startswith("scope_") for column in columns)
    SQLiteMemoryStore(path)


def test_schema_v1_is_rejected_without_mutating_the_database(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE memory_schema (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO memory_schema VALUES ('schema_version', '1')")
        connection.execute("CREATE TABLE old_data (text TEXT)")
        connection.execute("INSERT INTO old_data VALUES ('keep me')")
    before = path.read_bytes()
    with pytest.raises(IrisMemoryError, match="版本"):
        SQLiteMemoryStore(path)
    assert path.read_bytes() == before


def test_fts_initialization_failure_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError("FTS unavailable")
            return super().execute(sql, parameters)

    def connect(store: SQLiteMemoryStore) -> sqlite3.Connection:
        connection = sqlite3.connect(store.path, factory=FailingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(SQLiteMemoryStore, "_connect", connect)
    path = tmp_path / "failed.db"
    with pytest.raises(IrisMemoryError, match="初始化失败"):
        SQLiteMemoryStore(path)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )


def test_fts_execution_failure_does_not_fall_back_to_like(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE memory_items_fts")
    with pytest.raises(IrisMemoryError, match="搜索失败"):
        store.search(MemoryQuery(text="needle"))
