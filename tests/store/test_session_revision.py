"""Session revision 读取不物化历史或上下文投影。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisRunPersistenceError
from iris.lifecycle import LifecycleStore
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .test_lifecycle_store_contract import _prepare_tool


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """两个实现共用 revision-only 读取契约。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "revision.db")
    return InMemoryLifecycleStore()


def test_session_revision_matches_committed_history(store: LifecycleStore) -> None:
    """未创建 session 返回零，非空历史提交后返回当前 CAS revision。"""
    assert store.load_session_revision("missing") == 0
    committed = _prepare_tool(store)
    assert store.load_session_revision("session-1") == committed.session_revision == 1


def test_in_memory_revision_does_not_copy_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """整数读取不复制 store-owned 消息、摘要或窗口。"""
    store = InMemoryLifecycleStore()
    _prepare_tool(store)

    def reject_copy(value: object) -> None:
        pytest.fail("revision read must not copy session history")

    monkeypatch.setattr("iris.store.in_memory.deepcopy", reject_copy)
    assert store.load_session_revision("session-1") == 1
    assert store.load_session_revision("missing") == 0


def test_sqlite_revision_selects_only_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """查询只投影 revision，不读取或解析历史、摘要、窗口 JSON。"""
    store = SQLiteStore(tmp_path / "revision.db")
    _prepare_tool(store)
    reads: list[tuple[str, str]] = []
    connect = store._connect

    def recording_connect() -> sqlite3.Connection:
        connection = connect()

        def authorize(
            action: int, table: str, column: str, database: str, source: str | None
        ) -> int:
            if action == sqlite3.SQLITE_READ:
                reads.append((table, column))
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(store, "_connect", recording_connect)
    assert store.load_session_revision("session-1") == 1
    assert set(reads) == {("sessions", "revision"), ("sessions", "session_id")}


def test_sqlite_revision_preserves_raw_integer_validation(tmp_path: Path) -> None:
    """窄读取仍在 durable row 边界拒绝非整数 revision。"""
    store = SQLiteStore(tmp_path / "revision.db")
    _prepare_tool(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE sessions SET revision = 1.5 WHERE session_id = 'session-1'")
    with pytest.raises(IrisRunPersistenceError):
        store.load_session_revision("session-1")
