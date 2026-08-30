"""Lifecycle SQLite command transaction 的 rollback 测试。"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import iris.store.sqlite as sqlite_module
from iris.exceptions import IrisRunPersistenceError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    ClaimToolCall,
    CommitModelStep,
    CreateRun,
    FinishRun,
    RecoverActiveRun,
    RecoveryDisposition,
    ReserveModelStep,
    RunCheckpoint,
    RunCommit,
    RunErrorInfo,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SuspendRun,
)
from iris.message import Msg, ToolUseBlock
from iris.store import SQLiteStore

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_TOOL_FINGERPRINT = "a" * 64
_INTERACTION_ID = "int_" + "1" * 32


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


def test_statement_failure_rolls_back_the_complete_suspend_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suspend 任一 mutation statement 失败都不能暴露半提交 facts。"""
    original = sqlite_module._execute
    statement_count = 0
    count_store = SQLiteStore(tmp_path / "count-suspend.db")
    created = count_store.create_run(_create_command())

    def count_statements(
        connection: sqlite3.Connection,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> sqlite3.Cursor:
        nonlocal statement_count
        statement_count += 1
        return original(connection, sql, params)

    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite_module, "_execute", count_statements)
        count_store.suspend_run(_suspend_command(created.run.revision))

    for fail_at in range(1, statement_count + 1):
        path = tmp_path / f"suspend-failure-{fail_at}.db"
        store = SQLiteStore(path)
        created = store.create_run(_create_command())
        run_before = store.load_run("run-1")
        session_before = store.load_session("session-1")
        events_before = store.list_events("run-1")
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
                store.suspend_run(_suspend_command(created.run.revision))

        assert path.read_bytes() == before
        reopened = SQLiteStore(path)
        assert reopened.load_run("run-1") == run_before
        assert reopened.load_session("session-1") == session_before
        assert reopened.load_interaction(_INTERACTION_ID) is None
        assert reopened.list_tool_calls("run-1") == []
        assert reopened.list_events("run-1") == events_before


def test_statement_failure_rolls_back_outcome_unknown_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery 任一写入失败都必须保留旧 fence、claim、lane 与 events。"""
    original = sqlite_module._execute
    statement_count = 0
    count_store = SQLiteStore(tmp_path / "count-recovery.db")
    count_store.create_run(_create_command())
    claimed = _prepare_claimed_tool(count_store)

    def count_statements(
        connection: sqlite3.Connection,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> sqlite3.Cursor:
        nonlocal statement_count
        statement_count += 1
        return original(connection, sql, params)

    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite_module, "_execute", count_statements)
        count_store.recover_active_run(_recovery_command(claimed.run.revision, 2))

    for fail_at in range(1, statement_count + 1):
        path = tmp_path / f"recovery-failure-{fail_at}.db"
        store = SQLiteStore(path)
        store.create_run(_create_command())
        claimed = _prepare_claimed_tool(store)
        run_before = store.load_run("run-1")
        session_before = store.load_session("session-1")
        checkpoint_before = store.load_checkpoint("run-1")
        calls_before = store.list_tool_calls("run-1")
        events_before = store.list_events("run-1")
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
                store.recover_active_run(
                    _recovery_command(
                        claimed.run.revision,
                        claimed.checkpoint.sequence,
                    )
                )

        assert path.read_bytes() == before
        reopened = SQLiteStore(path)
        assert reopened.load_run("run-1") == run_before
        assert reopened.load_session("session-1") == session_before
        assert reopened.load_checkpoint("run-1") == checkpoint_before
        assert reopened.list_tool_calls("run-1") == calls_before
        assert reopened.list_events("run-1") == events_before
        with sqlite3.connect(path) as connection:
            activation = connection.execute(
                "SELECT status, outcome FROM run_activations WHERE activation_id = ?",
                ("act-1",),
            ).fetchone()
            lane = connection.execute(
                "SELECT run_id FROM session_run_lanes WHERE session_id = ?",
                ("session-1",),
            ).fetchone()
        assert activation == ("active", None)
        assert lane == ("run-1",)


def test_partial_session_message_insert_rolls_back_complete_model_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delta 中途插入失败时 metadata、history、run、checkpoint 与 event 一起回滚。"""
    path = tmp_path / "message-insert-failure.db"
    store = SQLiteStore(path)
    created = store.create_run(_create_command())
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="act-1",
            now=_NOW,
        )
    )
    run_before = store.load_run("run-1")
    session_before = store.load_session("session-1")
    checkpoint_before = store.load_checkpoint("run-1")
    events_before = store.list_events("run-1")
    command = CommitModelStep(
        run_id="run-1",
        expected_run_revision=reserved.run.revision,
        activation_id="act-1",
        expected_session_revision=0,
        message_delta=[Msg.user("one"), Msg.assistant("two")],
        usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
        checkpoint=RunCheckpoint(
            run_id="run-1",
            sequence=2,
            activation_id="act-1",
            engine_cursor={"position": "after_model", "step_index": 1},
            session_revision=1,
            model_steps_reserved=1,
            model_steps_committed=1,
            environment_fingerprint="environment-v1",
        ),
        assistant_message=Msg.assistant("two"),
        now=_NOW,
    )
    original = sqlite_module._executemany

    def insert_one_then_fail(
        connection: sqlite3.Connection,
        sql: str,
        params: list[tuple[object, ...]],
    ) -> sqlite3.Cursor:
        first, *_ = params
        connection.execute(sql, first)
        raise sqlite3.OperationalError("injected session message insert failure")

    monkeypatch.setattr(sqlite_module, "_executemany", insert_one_then_fail)
    with pytest.raises(IrisRunPersistenceError):
        store.commit_model_step(command)
    monkeypatch.setattr(sqlite_module, "_executemany", original)

    assert store.load_run("run-1") == run_before
    assert store.load_session("session-1") == session_before
    assert store.load_checkpoint("run-1") == checkpoint_before
    assert store.list_events("run-1") == events_before
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_messages").fetchone() == (0,)


@pytest.mark.parametrize("operation", ["finish", "recovery"])
def test_partial_session_message_insert_rolls_back_terminal_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Terminal closure 的 message row 写入后失败时，全部 aggregate facts 回滚。"""
    path = tmp_path / f"{operation}-message-insert-failure.db"
    store = SQLiteStore(path)
    store.create_run(_create_command())
    claimed = _prepare_claimed_tool(store)
    run_before = store.load_run("run-1")
    session_before = store.load_session("session-1")
    checkpoint_before = store.load_checkpoint("run-1")
    calls_before = store.list_tool_calls("run-1")
    events_before = store.list_events("run-1")
    result_before = store.load_result("run-1")
    lane_before = store.load_session_lane("session-1")
    with sqlite3.connect(path) as connection:
        activation_before = connection.execute(
            "SELECT status, outcome FROM run_activations WHERE activation_id = 'act-1'"
        ).fetchone()
        message_rows_before = connection.execute(
            """SELECT session_id, ordinal, message_json FROM session_messages
            ORDER BY session_id, ordinal"""
        ).fetchall()

    def insert_one_then_fail(
        connection: sqlite3.Connection,
        sql: str,
        params: list[tuple[object, ...]],
    ) -> sqlite3.Cursor:
        first, *_ = params
        connection.execute(sql, first)
        raise sqlite3.OperationalError("injected terminal session message insert failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite_module, "_executemany", insert_one_then_fail)
        with pytest.raises(IrisRunPersistenceError):
            if operation == "finish":
                store.finish_run(
                    FinishRun(
                        run_id="run-1",
                        expected_run_revision=claimed.run.revision,
                        activation_id="act-1",
                        stop_reason=RunStopReason.OUTCOME_UNKNOWN,
                        error=RunErrorInfo(
                            code="TOOL_OUTCOME_UNKNOWN",
                            message="工具结果不可证明",
                            source="tool",
                        ),
                        now=_NOW + timedelta(seconds=1),
                    )
                )
            else:
                store.recover_active_run(
                    _recovery_command(
                        claimed.run.revision,
                        claimed.checkpoint.sequence,
                    )
                )

    assert store.load_run("run-1") == run_before
    assert store.load_session("session-1") == session_before
    assert store.load_checkpoint("run-1") == checkpoint_before
    assert store.list_tool_calls("run-1") == calls_before
    assert store.list_events("run-1") == events_before
    assert store.load_result("run-1") == result_before
    assert store.load_session_lane("session-1") == lane_before
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute(
                "SELECT status, outcome FROM run_activations WHERE activation_id = 'act-1'"
            ).fetchone()
            == activation_before
        )
        assert (
            connection.execute(
                """SELECT session_id, ordinal, message_json FROM session_messages
            ORDER BY session_id, ordinal"""
            ).fetchall()
            == message_rows_before
        )


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


def _suspend_command(run_revision: int) -> SuspendRun:
    prepared = RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-question",
        tool_name="ask_question",
        arguments={"question": "继续吗？"},
        fingerprint=_TOOL_FINGERPRINT,
        phase="prepared",
        version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    interaction = HumanInteraction(
        interaction_id=_INTERACTION_ID,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        tool_call_id="call-question",
        status=InteractionStatus.PENDING,
        request=HumanInteractionRequest(
            tool_call=ToolCallSnapshot(
                tool_call_id="call-question",
                tool_name="ask_question",
                arguments={"question": "继续吗？"},
                workspace_root="workspace",
                fingerprint=_TOOL_FINGERPRINT,
            ),
            prompt=QuestionPrompt(question="继续吗？"),
        ),
        expires_at=_NOW + timedelta(minutes=5),
        created_at=_NOW,
    )
    checkpoint = RunCheckpoint(
        run_id="run-1",
        sequence=2,
        activation_id="act-1",
        engine_cursor={"position": "after_model", "step_index": 0},
        session_revision=0,
        model_steps_reserved=0,
        model_steps_committed=0,
        environment_fingerprint="environment-v1",
    )
    return SuspendRun(
        run_id="run-1",
        expected_run_revision=run_revision,
        activation_id="act-1",
        expected_session_revision=0,
        prepared_tool_calls=[prepared],
        checkpoint=checkpoint,
        pending_interaction=interaction,
        usage=RunUsage(),
        now=_NOW,
    )


def _prepare_claimed_tool(store: SQLiteStore) -> RunCommit:
    run = store.load_run("run-1")
    assert run is not None
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="act-1",
            now=_NOW,
        )
    )
    assistant = Msg.assistant([ToolUseBlock(id="call-tool", name="probe", input={"value": "A"})])
    prepared = RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-tool",
        tool_name="probe",
        arguments={"value": "A"},
        fingerprint=_TOOL_FINGERPRINT,
        phase="prepared",
        version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    committed = store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="act-1",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=[prepared],
            checkpoint=RunCheckpoint(
                run_id="run-1",
                sequence=2,
                activation_id="act-1",
                engine_cursor={"position": "after_model", "step_index": 1},
                session_revision=1,
                model_steps_reserved=1,
                model_steps_committed=1,
                environment_fingerprint="environment-v1",
            ),
            assistant_message=assistant,
            now=_NOW,
        )
    )
    return store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=committed.run.revision,
            activation_id="act-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_NOW,
        )
    )


def _recovery_command(run_revision: int, checkpoint_sequence: int) -> RecoverActiveRun:
    return RecoverActiveRun(
        run_id="run-1",
        expected_run_revision=run_revision,
        expected_activation_id="act-1",
        expected_checkpoint_sequence=checkpoint_sequence,
        recovery_disposition=RecoveryDisposition.OUTCOME_UNKNOWN,
        now=_NOW,
    )
