"""Lifecycle 整库备份保留身份和独立可写副本。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisRunPersistenceError
from iris.store import SQLiteStore

from .test_lifecycle_store_contract import _complete_history_turn


def test_backup_preserves_complete_runs_and_source(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "live.sqlite")
    first = _complete_history_turn(store, run_id="a", session_id="first")
    second = _complete_history_turn(store, run_id="b", session_id="second")
    backup = tmp_path / "snapshot/lifecycle.sqlite"
    store.backup(backup)
    saved = SQLiteStore(backup)
    assert saved.source_id == store.source_id
    assert saved.load_run("a") == first
    assert saved.load_run("b") == second
    assert saved.load_session("first") == store.load_session("first")
    _complete_history_turn(store, run_id="c", session_id="first")
    assert saved.load_session("first").revision < store.load_session("first").revision


def test_backup_rejects_source_path(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "live.sqlite")
    with pytest.raises(IrisRunPersistenceError):
        store.backup(store.path)
