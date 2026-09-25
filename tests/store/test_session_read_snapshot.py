"""完整会话在同一事务取原始行，锁外解码仍保持快照一致性。"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import iris.store.sqlite as sqlite_module
from iris.exceptions import IrisRunPersistenceError
from iris.message import Msg
from iris.store import SQLiteStore

from .test_run_input_commit import _input_command


def test_session_decoding_releases_lock_and_read_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """阻塞已读行的解码时，同一实例读取及另一连接的写入都能完成。"""
    store = SQLiteStore(tmp_path / "session.db")
    command = _input_command(store)
    started, release = Event(), Event()
    original = sqlite_module.decode_session_messages

    def decode(
        rows: Sequence[sqlite3.Row],
        *,
        expected_count: int,
        path: Path,
        operation: str,
        start_count: int = 0,
    ) -> list[Msg]:
        started.set()
        assert release.wait(5)
        return original(
            rows,
            expected_count=expected_count,
            path=path,
            operation=operation,
            start_count=start_count,
        )

    monkeypatch.setattr(sqlite_module, "decode_session_messages", decode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(store.load_session, "session")
        try:
            assert started.wait(2)
            assert pool.submit(store.load_session_revision, "session").result(2) == 0
            # 非 WAL 数据库也应在解码期间允许独立 writer 提交。
            writer = SQLiteStore(store.path)
            committed = pool.submit(writer.commit_run_input, command).result(2)
            assert committed.session_revision == 1
        finally:
            release.set()
        snapshot = reading.result(2)
    assert snapshot.revision == 0 and snapshot.messages == []
    assert store.load_session("session").messages == command.message_delta


def test_session_metadata_and_message_rows_share_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """metadata 读取后外部提交新消息，当前读取仍返回旧版本的完整组合。"""
    store = SQLiteStore(tmp_path / "snapshot.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
    writer = SQLiteStore(store.path)
    command = _input_command(store)
    original = store._select_session_metadata
    advanced = False

    def metadata(
        connection: sqlite3.Connection, session_id: str, *, operation: str
    ) -> sqlite_module._SessionMetadata | None:
        nonlocal advanced
        result = original(connection, session_id, operation=operation)
        if not advanced:
            advanced = True
            writer.commit_run_input(command)
        return result

    monkeypatch.setattr(store, "_select_session_metadata", metadata)
    first = store.load_session("session")
    second = store.load_session("session")
    assert first.revision == 0 and first.messages == [] and first.context_window is None
    assert second.revision == 1 and second.messages == command.message_delta
    assert second.context_window == command.initial_context_window


def test_lock_free_decode_retains_corrupt_message_error(tmp_path: Path) -> None:
    """锁外解析仍在 durable 边界报告损坏消息，不返回部分历史。"""
    store = SQLiteStore(tmp_path / "corrupt.db")
    store.commit_run_input(_input_command(store))
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE session_messages SET message_json = 'invalid' WHERE ordinal = 1")
    with pytest.raises(IrisRunPersistenceError) as error:
        store.load_session("session")
    assert error.value.context["operation"] == "load_session"
