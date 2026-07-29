"""Lifecycle SQLite command transaction 的 rollback 测试。"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import iris.store.sqlite as sqlite_module
from iris.exceptions import IrisRunPersistenceError
from iris.lifecycle import AgentRunOptions, AgentRunRequest, CreateRun, RunCheckpoint
from iris.store import SQLiteStore

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


def test_statement_failure_rolls_back_the_complete_create_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sqlite_module._execute
    statement_count = 0

    def count_statements(
        connection: sqlite3.Connection,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> sqlite3.Cursor:
        nonlocal statement_count
        statement_count += 1
        return original(connection, sql, params)

    count_path = tmp_path / "count.db"
    count_store = SQLiteStore(count_path)
    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite_module, "_execute", count_statements)
        count_store.create_run(_create_command())

    for fail_at in range(1, statement_count + 1):
        path = tmp_path / f"failure-{fail_at}.db"
        store = SQLiteStore(path)
        before = path.read_bytes()
        calls = 0

        def fail_statement(
            connection: sqlite3.Connection,
            sql: str,
            params: tuple[object, ...] = (),
            failure_point: int = fail_at,
        ) -> sqlite3.Cursor:
            nonlocal calls
            calls += 1
            if calls == failure_point:
                raise sqlite3.OperationalError("injected statement failure")
            return original(connection, sql, params)

        with monkeypatch.context() as patcher:
            patcher.setattr(sqlite_module, "_execute", fail_statement)
            with pytest.raises(IrisRunPersistenceError):
                store.create_run(_create_command())

        assert path.read_bytes() == before
        reopened = SQLiteStore(path)
        assert reopened.load_run("run-1") is None
        assert reopened.load_session("session-1").revision == 0


def _create_command() -> CreateRun:
    checkpoint = RunCheckpoint(
        run_id="run-1",
        sequence=1,
        activation_id="act-1",
        engine_cursor={"position": "before_model", "step_index": 0},
        session_revision=0,
        model_steps_reserved=0,
        model_steps_committed=0,
        environment_fingerprint="environment-v1",
    )
    return CreateRun(
        request=AgentRunRequest(
            input="start",
            session_id="session-1",
            run_id="run-1",
        ),
        options=AgentRunOptions(),
        agent_id="agent-1",
        environment_fingerprint="environment-v1",
        start_activation_id="act-1",
        initial_checkpoint=checkpoint,
        now=_NOW,
    )
