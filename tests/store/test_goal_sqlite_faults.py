"""Goal SQLite 原子写入、重新打开和读取边界故障测试。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisGoalPersistenceError
from iris.goal.models import GoalRef
from iris.goal.store import AdmitGoalRun, CreateGoal
from iris.lifecycle import AgentRunOptions
from iris.store import SQLiteStore
from iris.store import sqlite as sqlite_module

from ..goal.test_report_settlement import _admit, _finish, _report, _step
from .test_lifecycle_store_contract import _NOW, _create_command


def _goal_command() -> CreateGoal:
    return CreateGoal(
        goal_id="goal-1",
        session_id="session-1",
        objective="修复故障并验证",
        max_rounds=3,
        run_options=AgentRunOptions(),
        now=_NOW,
    )


def _counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name != 'lifecycle_schema'"
            )
        ]
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }


def test_goal_creation_failure_rolls_back_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目标插入失败时，同事务新增的空会话也消失。"""
    store = SQLiteStore(tmp_path / "create-fault.db")
    execute = sqlite_module._execute

    def fail_goal(
        connection: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()
    ) -> sqlite3.Cursor:
        if "INSERT INTO goals" in sql:
            raise sqlite3.OperationalError("injected goal insert failure")
        return execute(connection, sql, params)

    monkeypatch.setattr(sqlite_module, "_execute", fail_goal)
    with pytest.raises(IrisGoalPersistenceError) as captured:
        store.create_goal(_goal_command())
    assert captured.value.context["operation"] == "create_goal"
    assert not any(_counts(store.path).values())


@pytest.mark.parametrize(
    "statement",
    ["INSERT INTO run_events", "INSERT INTO goal_runs", "UPDATE goals"],
)
def test_goal_admission_failure_rolls_back_every_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str
) -> None:
    """普通 run 创建和目标绑定、计数任一处失败都不留下半次准入。"""
    store = SQLiteStore(tmp_path / "admit-fault.db")
    goal = store.create_goal(_goal_command())
    before = _counts(store.path)
    execute = sqlite_module._execute

    def fail_statement(
        connection: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()
    ) -> sqlite3.Cursor:
        if statement in sql:
            raise sqlite3.OperationalError("injected admission failure")
        return execute(connection, sql, params)

    monkeypatch.setattr(sqlite_module, "_execute", fail_statement)
    with pytest.raises(IrisGoalPersistenceError):
        store.admit_goal_run(
            AdmitGoalRun(
                expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
                create_run=_create_command(),
            )
        )
    assert _counts(store.path) == before
    assert store.get_current_goal("session-1") == goal
    assert store.load_session_lane("session-1") is None
    assert store.get_goal_run("run-1") is None


def test_goal_serialization_failure_does_not_create_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目标编码失败也必须回滚已经在同事务产生的普通执行事实。"""
    store = SQLiteStore(tmp_path / "encode-fault.db")
    goal = store.create_goal(_goal_command())
    before = _counts(store.path)
    dump = sqlite_module._dump_json
    options_encoded = 0

    def fail_options(value: object) -> str:
        nonlocal options_encoded
        if isinstance(value, AgentRunOptions):
            options_encoded += 1
            if options_encoded == 2:
                raise ValueError("injected goal options encoding failure")
        return dump(value)

    monkeypatch.setattr(sqlite_module, "_dump_json", fail_options)
    with pytest.raises(IrisGoalPersistenceError):
        store.admit_goal_run(
            AdmitGoalRun(
                expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
                create_run=_create_command(),
            )
        )
    assert options_encoded == 2
    assert _counts(store.path) == before
    assert store.get_goal(goal.goal_id) == goal


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("reason_json", "not-json"),
        ("run_options_json", '{"unexpected":true}'),
        ("created_at", "not-a-date"),
        ("objective", "   "),
    ],
)
def test_invalid_goal_row_is_goal_persistence_error(
    tmp_path: Path, column: str, value: str
) -> None:
    """持久化内容在读取边界验证，错误统一归属 Goal persistence。"""
    store = SQLiteStore(tmp_path / "corrupt-goal.db")
    goal = store.create_goal(_goal_command())
    with sqlite3.connect(store.path) as connection:
        connection.execute(f"UPDATE goals SET {column} = ?", (value,))
    reopened = SQLiteStore(store.path)
    with pytest.raises(IrisGoalPersistenceError) as captured:
        reopened.get_goal(goal.goal_id)
    assert captured.value.context["operation"] == "get_goal"
    assert captured.value.context["path"] == str(store.path)


def test_invalid_goal_binding_is_goal_persistence_error(tmp_path: Path) -> None:
    """报告标记与结算时间不一致的原始绑定必须在读取时拒绝。"""
    store = SQLiteStore(tmp_path / "corrupt-binding.db")
    goal = store.create_goal(_goal_command())
    store.admit_goal_run(AdmitGoalRun(expected=goal.ref, create_run=_create_command()))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE goal_runs SET applied_report_call_id = 'report' WHERE run_id = 'run-1'"
        )
    with pytest.raises(IrisGoalPersistenceError):
        store.get_goal_run("run-1")


def test_goal_binding_and_snapshot_survive_reopening(tmp_path: Path) -> None:
    """目标与普通 Run 共库持久化，重开保留计数和待结算绑定。"""
    store = SQLiteStore(tmp_path / "reopen-goal.db")
    goal = store.create_goal(_goal_command())
    admitted = store.admit_goal_run(
        AdmitGoalRun(
            expected=GoalRef(goal_id=goal.goal_id, revision=goal.revision),
            create_run=_create_command(),
        )
    )
    reopened = SQLiteStore(store.path)
    assert reopened.get_goal(goal.goal_id) == admitted.goal
    assert reopened.get_goal_run("run-1") == admitted.binding
    assert reopened.list_unsettled_goal_runs("session-1") == (admitted.binding,)
    assert reopened.load_session_revision("session-1") == 0


def test_settlement_binding_write_failure_rolls_back_goal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标完成写入后若绑定落盘失败，两项事实一起回滚并可重做。"""
    store = SQLiteStore(tmp_path / "settle-fault.db")
    current = _admit(store)
    current = _step(store, current, [("report_goal", {"goal_report": _report(store)}, False)])
    _finish(store, current)
    goal = store.get_goal("goal")
    binding = store.get_goal_run("run-1")
    execute = sqlite_module._execute

    def fail_binding(
        connection: sqlite3.Connection,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> sqlite3.Cursor:
        if "UPDATE goal_runs SET settled_at" in sql:
            raise sqlite3.OperationalError("injected settlement failure")
        return execute(connection, sql, params)

    monkeypatch.setattr(sqlite_module, "_execute", fail_binding)
    with pytest.raises(IrisGoalPersistenceError):
        store.settle_goal_run("run-1", now=_NOW)
    assert store.get_goal("goal") == goal
    assert store.get_goal_run("run-1") == binding
    monkeypatch.setattr(sqlite_module, "_execute", execute)
    assert store.settle_goal_run("run-1", now=_NOW).goal.status.value == "completed"
