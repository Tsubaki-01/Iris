"""Lifecycle SQLite 增量读写边界测试。

Example:
    UV_CACHE_DIR=/private/tmp/iris-uv-cache uv run pytest \
        tests/store/test_lifecycle_sqlite_incremental.py
"""

from __future__ import annotations

import ast
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.exceptions import IrisRunPersistenceError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    CreateRun,
    FinishRun,
    RecoverActiveRun,
    RecoveryDisposition,
    RequestCancellation,
    ReserveModelStep,
    ResolveInteraction,
    ResumeWaitingRun,
    RunCheckpoint,
    RunCommit,
    RunErrorInfo,
    RunToolCallRecord,
    RunUsage,
    SuspendRun,
)
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.store import SQLiteStore
from iris.tools import ToolResult

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_T1 = _NOW + timedelta(seconds=1)
_T2 = _NOW + timedelta(seconds=2)
_T3 = _NOW + timedelta(seconds=3)
_TOOL_FINGERPRINT = "a" * 64
_INTERACTION_ID = "int_" + "1" * 32
_BUSINESS_TABLES = {
    "agent_runs",
    "run_activations",
    "run_checkpoints",
    "run_events",
    "run_interactions",
    "run_tool_calls",
    "session_run_lanes",
    "sessions",
}


def _create(
    store: SQLiteStore,
    *,
    run_id: str,
    session_id: str,
    activation_id: str,
) -> None:
    """创建一个独立的 active run。"""
    checkpoint = RunCheckpoint(
        run_id=run_id,
        sequence=1,
        activation_id=activation_id,
        engine_cursor={"position": "before_model", "step_index": 0},
        session_revision=0,
        model_steps_reserved=0,
        model_steps_committed=0,
        environment_fingerprint="environment-v1",
    )
    store.create_run(
        CreateRun(
            request=AgentRunRequest(
                input="start",
                session_id=session_id,
                run_id=run_id,
            ),
            options=AgentRunOptions(),
            agent_id="agent-1",
            environment_fingerprint="environment-v1",
            start_activation_id=activation_id,
            initial_checkpoint=checkpoint,
            now=_NOW,
        )
    )


def _checkpoint(
    *,
    sequence: int,
    activation_id: str,
    session_revision: int,
    reserved: int = 0,
    committed: int = 0,
) -> RunCheckpoint:
    """构造 run-1 的测试 checkpoint。"""
    return RunCheckpoint(
        run_id="run-1",
        sequence=sequence,
        activation_id=activation_id,
        engine_cursor={"step_index": committed},
        session_revision=session_revision,
        model_steps_reserved=reserved,
        model_steps_committed=committed,
        environment_fingerprint="environment-v1",
    )


def _waiting_interaction() -> HumanInteraction:
    """构造绑定 ask_question 的 pending interaction。"""
    return HumanInteraction(
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
        created_at=_T1,
    )


def _suspend(store: SQLiteStore) -> RunCommit:
    """把 run-1 推进到 pending waiting。"""
    run = store.load_run("run-1")
    assert run is not None
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
        created_at=_T1,
        updated_at=_T1,
    )
    return store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                sequence=2,
                activation_id="activation-1",
                session_revision=0,
            ),
            pending_interaction=_waiting_interaction(),
            usage=run.usage,
            now=_T1,
        )
    )


def _suspend_and_resolve(store: SQLiteStore) -> None:
    """把 run-1 推进到 resolved waiting。"""
    waiting = _suspend(store)
    assert waiting.interaction is not None
    store.resolve_interaction(
        ResolveInteraction(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            interaction_id=_INTERACTION_ID,
            expected_interaction_version=waiting.interaction.version,
            response=QuestionInteractionResponse(answer="继续"),
            expected_fingerprint=_TOOL_FINGERPRINT,
            now=_T2,
        )
    )


def _prepare_tool(store: SQLiteStore) -> None:
    """为 run-1 持久化一条 prepared tool call。"""
    run = store.load_run("run-1")
    assert run is not None
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            now=_T1,
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
        created_at=_T1,
        updated_at=_T1,
    )
    store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                sequence=2,
                activation_id="activation-1",
                session_revision=1,
                reserved=1,
                committed=1,
            ),
            assistant_message=assistant,
            now=_T1,
        )
    )


def _capture_sql[ResultT](
    monkeypatch: pytest.MonkeyPatch,
    store: SQLiteStore,
    action: Callable[[], ResultT],
) -> tuple[ResultT, list[str]]:
    """记录一次公开操作在新连接上执行的规范化 SQL。"""
    statements: list[str] = []
    original_connect = store._connect

    def traced_connect() -> sqlite3.Connection:
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "_connect", traced_connect)
        result = action()
    normalized = [re.sub(r"\s+", " ", statement.strip()).lower() for statement in statements]
    return result, normalized


def _selected_tables(statements: list[str]) -> set[str]:
    """返回 trace 中被 SELECT 的 lifecycle 业务表。"""
    return {
        table
        for table in _BUSINESS_TABLES
        if any(f" from {table}" in f" {statement}" for statement in statements)
    }


def _contains_unqualified_business_delete(statements: list[str]) -> bool:
    """判断 mutation 是否清空任一 lifecycle 业务表。"""
    return any(
        statement == f"delete from {table}"
        for statement in statements
        for table in _BUSINESS_TABLES
    )


def test_single_entity_reads_query_only_the_target_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单实体 read 不应扫描无关 lifecycle 表。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")

    _, statements = _capture_sql(monkeypatch, store, lambda: store.load_run("run-1"))
    assert _selected_tables(statements) == {"agent_runs"}

    _, statements = _capture_sql(monkeypatch, store, lambda: store.load_session("session-1"))
    assert _selected_tables(statements) == {"sessions"}

    _, statements = _capture_sql(monkeypatch, store, lambda: store.load_checkpoint("run-1"))
    assert _selected_tables(statements) == {"run_checkpoints"}

    _, statements = _capture_sql(monkeypatch, store, lambda: store.load_interaction("missing"))
    assert _selected_tables(statements) == {"run_interactions"}


def test_collection_reads_query_only_run_and_requested_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """集合 read 只读取存在性与目标集合。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")

    _, statements = _capture_sql(monkeypatch, store, lambda: store.list_tool_calls("run-1"))
    assert _selected_tables(statements) == {"agent_runs", "run_tool_calls"}
    assert any("where run_id =" in statement for statement in statements)
    assert any("order by step_index, ordinal" in statement for statement in statements)

    _, statements = _capture_sql(monkeypatch, store, lambda: store.list_events("run-1", 1))
    assert _selected_tables(statements) == {"agent_runs", "run_events"}
    assert any("sequence >" in statement for statement in statements)


def test_active_result_read_is_write_free_and_only_queries_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active result read 不应加载 aggregate 或执行 DML。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")

    result, statements = _capture_sql(monkeypatch, store, lambda: store.load_result("run-1"))

    assert result is None
    assert _selected_tables(statements) == {"agent_runs"}
    assert not any(
        statement.startswith(("begin immediate", "insert ", "update ", "delete "))
        for statement in statements
    )


def test_waiting_result_read_only_queries_run_and_interaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting result 只读取目标 run 与 pending interaction。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _suspend(store)

    result, statements = _capture_sql(monkeypatch, store, lambda: store.load_result("run-1"))

    assert result is not None
    assert result.run.phase == "waiting"
    assert _selected_tables(statements) == {"agent_runs", "run_interactions"}
    assert not any(
        statement.startswith(("begin immediate", "insert ", "update ", "delete "))
        for statement in statements
    )


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        ("run", "load_run"),
        ("checkpoint", "load_checkpoint"),
        ("tool", "list_tool_calls"),
        ("interaction", "load_interaction"),
        ("event", "list_events"),
    ],
)
def test_corrupt_target_row_is_mapped_to_persistence_error(
    tmp_path: Path,
    kind: str,
    operation: str,
) -> None:
    """每类 target row 的反序列化失败都保留公开 operation context。"""
    path = tmp_path / f"{kind}.db"
    store = SQLiteStore(path)
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    if kind == "tool":
        _prepare_tool(store)
    elif kind == "interaction":
        _suspend(store)

    statements = {
        "run": "UPDATE agent_runs SET request_json = '[]' WHERE run_id = 'run-1'",
        "checkpoint": ("UPDATE run_checkpoints SET cursor_json = '[]' WHERE run_id = 'run-1'"),
        "tool": (
            "UPDATE run_tool_calls SET arguments_json = '[]' "
            "WHERE run_id = 'run-1' AND tool_call_id = 'call-tool'"
        ),
        "interaction": (
            "UPDATE run_interactions SET request_json = '[]' "
            f"WHERE interaction_id = '{_INTERACTION_ID}'"
        ),
        "event": (
            "UPDATE run_events SET payload_json = '[]' WHERE run_id = 'run-1' AND sequence = 1"
        ),
    }
    with sqlite3.connect(path) as connection:
        connection.execute(statements[kind])

    readers: dict[str, Callable[[], object]] = {
        "run": lambda: store.load_run("run-1"),
        "checkpoint": lambda: store.load_checkpoint("run-1"),
        "tool": lambda: store.list_tool_calls("run-1"),
        "interaction": lambda: store.load_interaction(_INTERACTION_ID),
        "event": lambda: store.list_events("run-1"),
    }
    with pytest.raises(IrisRunPersistenceError) as captured:
        readers[kind]()

    assert captured.value.context["operation"] == operation
    assert captured.value.context["path"] == str(path)


def test_create_run_does_not_rewrite_existing_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """创建独立 run 时不得删除并重建已有 aggregate。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    run_before = store.load_run("run-1")
    events_before = store.list_events("run-1")

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: _create(
            store,
            run_id="run-2",
            session_id="session-2",
            activation_id="activation-2",
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert store.load_run("run-1") == run_before
    assert store.list_events("run-1") == events_before


def test_reserve_model_step_updates_only_current_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model reservation 不得重写其他 run 或既有 events。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    run = store.load_run("run-1")
    unrelated_run = store.load_run("run-2")
    unrelated_events = store.list_events("run-2")
    assert run is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.reserve_model_step(
            ReserveModelStep(
                run_id="run-1",
                expected_run_revision=run.revision,
                activation_id="activation-1",
                now=_NOW,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert store.load_run("run-2") == unrelated_run
    assert store.list_events("run-2") == unrelated_events


def test_resume_waiting_run_updates_only_waiting_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume activation 不得重写无关 aggregate。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    _suspend_and_resolve(store)
    run = store.load_run("run-1")
    checkpoint = store.load_checkpoint("run-1")
    unrelated_run = store.load_run("run-2")
    assert run is not None
    assert checkpoint is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.resume_waiting_run(
            ResumeWaitingRun(
                run_id="run-1",
                expected_run_revision=run.revision,
                new_activation_id="activation-resume",
                kind="resume",
                expected_checkpoint_sequence=checkpoint.sequence,
                now=_T3,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert store.load_run("run-2") == unrelated_run


def test_commit_model_step_updates_only_current_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model commit 只更新当前 history/checkpoint/tool intents/event。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    run = store.load_run("run-1")
    assert run is not None
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            now=_T1,
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
        created_at=_T1,
        updated_at=_T1,
    )
    unrelated_run = store.load_run("run-2")
    unrelated_session = store.load_session("session-2")

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.commit_model_step(
            CommitModelStep(
                run_id="run-1",
                expected_run_revision=reserved.run.revision,
                activation_id="activation-1",
                expected_session_revision=0,
                message_delta=[assistant],
                usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
                prepared_tool_calls=[prepared],
                checkpoint=_checkpoint(
                    sequence=2,
                    activation_id="activation-1",
                    session_revision=1,
                    reserved=1,
                    committed=1,
                ),
                assistant_message=assistant,
                now=_T2,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert store.load_run("run-2") == unrelated_run
    assert store.load_session("session-2") == unrelated_session


def test_claim_tool_call_updates_exact_tool_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool claim 使用 run/tool/version CAS，且不重写无关 aggregate。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    _prepare_tool(store)
    run = store.load_run("run-1")
    unrelated_run = store.load_run("run-2")
    assert run is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.claim_tool_call(
            ClaimToolCall(
                run_id="run-1",
                expected_run_revision=run.revision,
                activation_id="activation-1",
                tool_call_id="call-tool",
                fingerprint=_TOOL_FINGERPRINT,
                expected_tool_version=1,
                now=_T2,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("update run_tool_calls")
        and "where run_id =" in statement
        and "tool_call_id =" in statement
        and "version =" in statement
        for statement in statements
    )
    assert store.load_run("run-2") == unrelated_run


def test_commit_tool_result_updates_exact_current_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool result 只更新目标 tool/session/run/checkpoint 并 append event。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    _prepare_tool(store)
    run = store.load_run("run-1")
    assert run is not None
    claimed = store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )
    unrelated_run = store.load_run("run-2")
    unrelated_session = store.load_session("session-2")

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.commit_tool_result(
            CommitToolResult(
                run_id="run-1",
                expected_run_revision=claimed.run.revision,
                activation_id="activation-1",
                expected_session_revision=1,
                tool_call_id="call-tool",
                expected_tool_version=2,
                result=ToolResult(
                    tool_use_id="call-tool",
                    tool_name="probe",
                    content=[TextBlock(text="done")],
                ),
                message_delta=[
                    Msg.tool_result(
                        tool_use_id="call-tool",
                        name="probe",
                        content="done",
                    )
                ],
                checkpoint=_checkpoint(
                    sequence=3,
                    activation_id="activation-1",
                    session_revision=2,
                    reserved=1,
                    committed=1,
                ),
                now=_T3,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("update run_tool_calls")
        and "where run_id =" in statement
        and "tool_call_id =" in statement
        and "version =" in statement
        for statement in statements
    )
    assert store.load_run("run-2") == unrelated_run
    assert store.load_session("session-2") == unrelated_session


def test_suspend_run_inserts_only_current_waiting_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suspend 只写当前 interaction、tool intents 和 aggregate rows。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    run = store.load_run("run-1")
    unrelated_run = store.load_run("run-2")
    assert run is not None
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
        created_at=_T1,
        updated_at=_T1,
    )

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.suspend_run(
            SuspendRun(
                run_id="run-1",
                expected_run_revision=run.revision,
                activation_id="activation-1",
                expected_session_revision=0,
                prepared_tool_calls=[prepared],
                checkpoint=_checkpoint(
                    sequence=2,
                    activation_id="activation-1",
                    session_revision=0,
                ),
                pending_interaction=_waiting_interaction(),
                usage=run.usage,
                now=_T1,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert sum(statement.startswith("insert into run_events") for statement in statements) == 1
    assert store.load_run("run-2") == unrelated_run


def test_resolve_interaction_updates_exact_versioned_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve 使用 interaction version CAS 且只 append 一个 event。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    run = store.load_run("run-1")
    assert run is not None
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
        created_at=_T1,
        updated_at=_T1,
    )
    waiting = store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                sequence=2,
                activation_id="activation-1",
                session_revision=0,
            ),
            pending_interaction=_waiting_interaction(),
            usage=run.usage,
            now=_T1,
        )
    )

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.resolve_interaction(
            ResolveInteraction(
                run_id="run-1",
                expected_run_revision=waiting.run.revision,
                interaction_id=_INTERACTION_ID,
                expected_interaction_version=1,
                response=QuestionInteractionResponse(answer="继续"),
                expected_fingerprint=_TOOL_FINGERPRINT,
                now=_T2,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("update run_interactions")
        and "where interaction_id =" in statement
        and "version =" in statement
        for statement in statements
    )
    assert sum(statement.startswith("insert into run_events") for statement in statements) == 1


def test_active_cancellation_updates_only_current_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active cancellation 只追加 intent，不释放 lane 或重写其他 aggregate。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    run = store.load_run("run-1")
    unrelated_run = store.load_run("run-2")
    assert run is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.request_cancellation(
            RequestCancellation(
                run_id="run-1",
                expected_run_revision=run.revision,
                activation_id="activation-1",
                reason="user requested",
                now=_T1,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert not any(statement.startswith("delete ") for statement in statements)
    assert store.load_run("run-2") == unrelated_run


def test_waiting_cancellation_deletes_only_exact_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting settlement 只关闭当前 interaction 并释放精确 lane。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _suspend_and_resolve(store)
    run = store.load_run("run-1")
    assert run is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.request_cancellation(
            RequestCancellation(
                run_id="run-1",
                expected_run_revision=run.revision,
                reason="user requested",
                settle_waiting=True,
                now=_T3,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("delete from session_run_lanes")
        and "where session_id =" in statement
        and "run_id =" in statement
        for statement in statements
    )


def test_finish_run_deletes_only_exact_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal finish 只 settle 当前 activation、run 和 lane。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    run = store.load_run("run-1")
    unrelated_run = store.load_run("run-2")
    assert run is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.finish_run(
            FinishRun(
                run_id="run-1",
                expected_run_revision=run.revision,
                activation_id="activation-1",
                stop_reason="completed",
                assistant_message=Msg.assistant("done"),
                now=_T1,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("delete from session_run_lanes")
        and "where session_id =" in statement
        and "run_id =" in statement
        for statement in statements
    )
    assert store.load_run("run-2") == unrelated_run


def test_outcome_unknown_finish_updates_only_claimed_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outcome-unknown finish 用 version CAS 闭合当前 run 的 claimed call。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _prepare_tool(store)
    run = store.load_run("run-1")
    assert run is not None
    claimed = store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.finish_run(
            FinishRun(
                run_id="run-1",
                expected_run_revision=claimed.run.revision,
                activation_id="activation-1",
                stop_reason="outcome_unknown",
                error=RunErrorInfo(
                    code="TOOL_OUTCOME_UNKNOWN",
                    message="工具结果不可证明",
                    source="tool",
                ),
                now=_T3,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("update run_tool_calls") and "version =" in statement
        for statement in statements
    )


def test_safe_recovery_rebinds_only_current_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe recovery 放弃旧 fence，并只为当前 run 创建新 activation。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    _create(store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    run = store.load_run("run-1")
    unrelated_run = store.load_run("run-2")
    assert run is not None

    _, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.recover_active_run(
            RecoverActiveRun(
                run_id="run-1",
                expected_run_revision=run.revision,
                expected_activation_id="activation-1",
                expected_checkpoint_sequence=1,
                recovery_disposition=RecoveryDisposition.RESUME,
                new_activation_id="activation-recovery",
                now=_T1,
            )
        ),
    )

    assert not _contains_unqualified_business_delete(statements)
    assert not any(statement.startswith("delete ") for statement in statements)
    assert store.load_run("run-2") == unrelated_run


def test_outcome_ready_finalize_deletes_only_exact_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outcome-ready recovery 只结算当前 run 并精确释放 lane。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store, run_id="run-1", session_id="session-1", activation_id="activation-1")
    run = store.load_run("run-1")
    assert run is not None
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    checkpoint = RunCheckpoint.model_validate(
        _checkpoint(
            sequence=2,
            activation_id="activation-1",
            session_revision=1,
            reserved=1,
            committed=1,
        ).model_dump()
        | {"resumability": "outcome_ready"}
    )
    committed = store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[Msg.assistant("done")],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            checkpoint=checkpoint,
            assistant_message=Msg.assistant("done"),
            now=_T2,
        )
    )

    terminal, statements = _capture_sql(
        monkeypatch,
        store,
        lambda: store.recover_active_run(
            RecoverActiveRun(
                run_id="run-1",
                expected_run_revision=committed.run.revision,
                expected_activation_id="activation-1",
                expected_checkpoint_sequence=checkpoint.sequence,
                recovery_disposition=RecoveryDisposition.FINALIZE,
                now=_T3,
            )
        ),
    )

    assert terminal.run.stop_reason == "completed"
    assert [event.kind for event in terminal.events] == [
        "activation.abandoned",
        "run.terminal",
    ]
    assert not _contains_unqualified_business_delete(statements)
    assert any(
        statement.startswith("delete from session_run_lanes")
        and "where session_id =" in statement
        and "run_id =" in statement
        for statement in statements
    )


def test_sqlite_store_has_no_snapshot_or_concrete_store_dependency() -> None:
    """SQLite 实现不得保留内存 store fallback 或全量 snapshot helpers。"""
    path = Path(__file__).parents[2] / "src" / "iris" / "store" / "sqlite.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert "in_memory" not in imported_modules
    assert not {"_load_memory", "_replace_all", "_mutate_snapshot"} & function_names
    assert "InMemoryLifecycleStore" not in source

    sql_literals = {
        re.sub(r"\s+", " ", node.value.strip()).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for sql in sql_literals:
        touched_tables = {table for table in _BUSINESS_TABLES if f" {table}" in f" {sql}"}
        if touched_tables and sql.startswith(("select ", "update ", "delete ")):
            assert " where " in f" {sql} "
        assert not sql.startswith(("update run_events ", "delete from run_events "))
