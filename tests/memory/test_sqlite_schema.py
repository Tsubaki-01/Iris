"""Memory schema v5 的 FTS、namespace 版本与初始化失败边界。"""

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory.models import MemorySearchQuery
from iris.memory.sqlite import SQLiteMemoryStore


def test_schema_v5_stores_namespaces_fts_and_revisions_and_reopens(tmp_path: Path) -> None:
    """新结构同时包含 FTS 与版本表，空 namespace 读取不创建状态行。"""
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path)
    assert store.read_namespace_state("unwritten").item_revision == 0
    assert store.read_namespace_snapshot("unwritten").items == ()
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM memory_schema WHERE key='schema_version'"
        ).fetchone()
        assert version == ("5",)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert {"memory_items_fts", "memory_namespace_state"} <= tables
        assert connection.execute("SELECT count(*) FROM memory_namespace_state").fetchone() == (0,)
        fts_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='memory_items_fts'"
        ).fetchone()[0]
        assert "unicode61 remove_diacritics 0" in fts_sql
        for table in ["memory_items", "memory_observations", "memory_episodes", "memory_events"]:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            assert "namespace" in columns
            assert not any(column.startswith("scope_") for column in columns)
    SQLiteMemoryStore(path)


@pytest.mark.parametrize("version", ["1", "2", "3", "4", "999", None])
def test_old_or_missing_schema_version_is_rejected_without_mutating_database(
    tmp_path: Path, version: str | None
) -> None:
    """旧版、未知版本和缺失版本行不会触发迁移或结构写入。"""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE memory_schema (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if version is not None:
            connection.execute("INSERT INTO memory_schema VALUES ('schema_version', ?)", (version,))
        connection.execute("CREATE TABLE old_data (text TEXT)")
        connection.execute("INSERT INTO old_data VALUES ('keep me')")
    before = path.read_bytes()
    with pytest.raises(IrisMemoryError, match="版本"):
        SQLiteMemoryStore(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "statement", ["CREATE VIRTUAL TABLE", "CREATE TABLE memory_namespace_state"]
)
def test_schema_initialization_failure_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str
) -> None:
    """FTS 或 namespace 状态表创建失败时回滚整个新库初始化。"""

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            if statement in sql:
                raise sqlite3.OperationalError("schema unavailable")
            return super().execute(sql, parameters)

    def connect(store: SQLiteMemoryStore) -> sqlite3.Connection:
        connection = sqlite3.connect(store.path, factory=FailingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(SQLiteMemoryStore, "_connect", connect)
    path = tmp_path / "failed.db"
    with pytest.raises(IrisMemoryError, match="失败"):
        SQLiteMemoryStore(path)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )


def test_fts_execution_failure_does_not_fall_back_to_like(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE memory_items_fts")
    with pytest.raises(IrisMemoryError, match="失败"):
        store.search(MemorySearchQuery(query="needle"), ["project"])
