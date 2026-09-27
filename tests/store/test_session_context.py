"""有效历史快照只读取后缀与任务锚点，原文和派生状态原子推进。"""

import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

import iris.store.sqlite as sqlite_module
from iris.exceptions import IrisRunConflictError, IrisRunPersistenceError
from iris.lifecycle import (
    AgentRunOptions,
    FinishRun,
    ForkSession,
    LifecycleStore,
    ReserveModelStep,
    RunCommit,
    RunLimits,
    RunPhase,
    RunStopReason,
    SessionReadState,
)
from iris.message import Msg
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .test_lifecycle_store_contract import _compaction_command, _create_command
from .test_run_input_commit import NOW, _input_command
from .test_session_projection import _replay, _result


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """两种存储执行相同有效历史契约。"""
    return (
        SQLiteStore(tmp_path / "context.db")
        if request.param == "sqlite"
        else InMemoryLifecycleStore()
    )


def _history(count: int = 20, *, bci: bool = True) -> list[Msg]:
    messages = [Msg.user("input"), _result("a", "b")]
    if bci:
        messages.insert(
            0, Msg.user("BCI", sender="context", metadata={"context_kind": "before_current_input"})
        )
    messages.extend(Msg.assistant(f"old {index}") for index in range(count - len(messages) - 3))
    messages.extend([Msg.user("steer"), Msg.user("context", sender="context"), _result("c", "a")])
    return messages


def _seed(store: LifecycleStore, messages: list[Msg], covered: int) -> RunCommit:
    command = replace(_input_command(store), message_delta=messages)
    committed = store.commit_run_input(command)
    if not covered:
        return committed
    ready = store.reserve_model_step(
        ReserveModelStep(
            run_id=command.run_id,
            expected_run_revision=committed.run.revision,
            activation_id=command.activation_id,
            now=NOW,
        )
    )
    return store.commit_compaction(_compaction_command(ready, count=covered))


@pytest.mark.parametrize("bci", [False, True])
@pytest.mark.parametrize("covered", [0, 1, 2, 17, 20])
def test_context_preserves_absolute_anchors_and_independent_objects(
    store: LifecycleStore,
    bci: bool,
    covered: int,
) -> None:
    messages = _history(bci=bci)
    committed = _seed(store, messages, covered)
    snapshot = store.load_run_context(committed.run.run_id, include_tool_discovery=True)
    expected = (0, 1, 17) if bci else (0, 17)
    assert snapshot.header.message_count == 20
    assert snapshot.raw_tail == tuple(messages[covered:])
    assert snapshot.protected_indices == expected
    assert snapshot.protected_prefix_messages == tuple(
        (index, messages[index]) for index in expected if index < covered
    )
    assert snapshot.tool_discovery == _replay(messages).tool_discovery
    assert snapshot.header == store.load_session_header("session")
    assert (
        store.load_run_context(committed.run.run_id, include_tool_discovery=False).tool_discovery
        is None
    )
    snapshot.tool_discovery.discovered_at["injected"] = 999
    sample = snapshot.raw_tail[0] if snapshot.raw_tail else snapshot.protected_prefix_messages[0][1]
    sample.metadata["injected"] = True
    again = store.load_run_context(committed.run.run_id, include_tool_discovery=True)
    assert again.tool_discovery == _replay(messages).tool_discovery
    assert store.load_session("session").messages == messages
    if isinstance(store, SQLiteStore):
        assert (
            SQLiteStore(store.path).load_run_context(
                committed.run.run_id, include_tool_discovery=True
            )
            == again
        )


def test_failed_checkpoint_does_not_publish_candidate_projection(store: LifecycleStore) -> None:
    command = replace(_input_command(store), message_delta=[Msg.user("input"), _result("a")])
    command = replace(command, checkpoint=command.checkpoint.model_copy(update={"sequence": 7}))
    with pytest.raises(IrisRunConflictError):
        store.commit_run_input(command)
    snapshot = store.load_run_context(command.run_id, include_tool_discovery=True)
    assert snapshot.header.message_count == 0
    assert snapshot.protected_indices == ()
    assert snapshot.tool_discovery == SessionReadState().tool_discovery


def test_fork_replays_cutoff_and_branches_remain_independent(store: LifecycleStore) -> None:
    messages = _history()
    current = _seed(store, messages, 17)
    finished = store.finish_run(
        FinishRun(
            run_id=current.run.run_id,
            expected_run_revision=current.run.revision,
            activation_id=current.run.current_activation_id,
            stop_reason=RunStopReason.COMPLETED,
            now=NOW,
        )
    )
    second = replace(
        _input_command(store, run_id="second"),
        message_delta=[Msg.user("later"), _result("parent")],
        initial_context_window=None,
    )
    store.commit_run_input(second)
    fork = store.fork_session(
        ForkSession(source_run_id=finished.run.run_id, target_session_id="branch", now=NOW)
    )
    assert fork.revision == 0 and fork.context_window is None
    child_command = _input_command(store, run_id="child", session_id="branch")
    child_context = store.load_run_context("child", include_tool_discovery=True)
    assert child_context.tool_discovery == _replay(messages).tool_discovery
    assert child_context.protected_indices == ()
    child_delta = [Msg.user("child"), _result("branch")]
    store.commit_run_input(replace(child_command, message_delta=child_delta))
    assert (
        "branch"
        not in store.load_run_context(
            "second", include_tool_discovery=True
        ).tool_discovery.discovered_at
    )
    assert (
        "parent"
        not in store.load_run_context(
            "child", include_tool_discovery=True
        ).tool_discovery.discovered_at
    )


@pytest.mark.parametrize("count", [20, 2000])
def test_sqlite_context_decode_is_bounded_and_header_excludes_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    store = SQLiteStore(tmp_path / "bounds.db")
    messages = _history(count)
    _seed(store, messages, count - 2)
    decoded: list[int] = []
    queries: list[str] = []
    original_decode = sqlite_module.decode_session_messages
    original_single = sqlite_module.decode_session_message
    original_connect = store._connect

    def connect() -> sqlite3.Connection:
        connection = original_connect()
        connection.set_trace_callback(queries.append)
        return connection

    def decode(
        rows: Sequence[sqlite3.Row],
        *,
        expected_count: int,
        path: Path,
        operation: str,
        start_count: int = 0,
    ) -> list[Msg]:
        decoded.extend(row["ordinal"] - 1 for row in rows)
        return original_decode(
            rows,
            expected_count=expected_count,
            path=path,
            operation=operation,
            start_count=start_count,
        )

    def single(row: sqlite3.Row, *, path: Path, operation: str) -> Msg:
        decoded.append(row["ordinal"] - 1)
        return original_single(row, path=path, operation=operation)

    monkeypatch.setattr(store, "_connect", connect)
    monkeypatch.setattr(sqlite_module, "decode_session_messages", decode)
    monkeypatch.setattr(sqlite_module, "decode_session_message", single)
    header = store.load_session_header("session")
    assert header.message_count == count
    assert not any(
        "session_messages" in query or "tool_discovery_json" in query or "compaction_json" in query
        for query in queries
    )
    queries.clear()
    store.load_run_context("run-input", include_tool_discovery=False)
    assert sorted(decoded) == [0, 1, count - 3, count - 2, count - 1]
    assert not any("tool_discovery_json" in query for query in queries)
    assert not any("SELECT * FROM agent_runs" in query for query in queries)


def test_sqlite_append_statement_failure_rolls_back_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(tmp_path / "rollback.db")
    command = replace(_input_command(store), message_delta=[Msg.user("input"), _result("a")])
    original = sqlite_module._executemany

    def fail(
        connection: sqlite3.Connection, sql: str, params: list[tuple[object, ...]]
    ) -> sqlite3.Cursor:
        original(connection, sql, params)
        raise sqlite3.OperationalError("after messages")

    monkeypatch.setattr(sqlite_module, "_executemany", fail)
    with pytest.raises(IrisRunPersistenceError):
        store.commit_run_input(command)
    snapshot = SQLiteStore(store.path).load_run_context(command.run_id, include_tool_discovery=True)
    assert snapshot.header.message_count == 0
    assert snapshot.tool_discovery == SessionReadState().tool_discovery


def test_sqlite_context_rows_and_projection_share_metadata_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """另一连接在 metadata 后提交时，本次后缀和发现仍属于旧版本。"""
    store = SQLiteStore(tmp_path / "consistent.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
    writer = SQLiteStore(store.path)
    command = replace(_input_command(store), message_delta=[Msg.user("input"), _result("a")])
    original = store._select_session_metadata
    advanced = False

    def metadata(
        connection: sqlite3.Connection,
        session_id: str,
        *,
        operation: str,
    ) -> sqlite_module._SessionMetadata | None:
        nonlocal advanced
        result = original(connection, session_id, operation=operation)
        if not advanced:
            advanced = True
            writer.commit_run_input(command)
        return result

    monkeypatch.setattr(store, "_select_session_metadata", metadata)
    before = store.load_run_context(command.run_id, include_tool_discovery=True)
    after = store.load_run_context(command.run_id, include_tool_discovery=True)
    assert before.header.revision == before.header.message_count == 0
    assert before.raw_tail == before.protected_indices == ()
    assert before.tool_discovery == SessionReadState().tool_discovery
    assert after.header.revision == 1 and after.header.message_count == 2
    assert after.raw_tail == tuple(command.message_delta)
    assert after.protected_indices == (0,)
    assert after.tool_discovery == _replay(command.message_delta).tool_discovery


def test_sqlite_empty_input_and_control_do_not_parse_discovery(tmp_path: Path) -> None:
    """不消费发现状态的读取和空 delta 不因该 JSON 被读取而产生额外依赖。"""
    store = SQLiteStore(tmp_path / "narrow.db")
    command = replace(_input_command(store), message_delta=[])
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE sessions SET tool_discovery_json = 'not json'")
    assert store.load_session_header("session").message_count == 0
    assert store.load_run_control(command.run_id) is not None
    assert (
        store.load_run_context(command.run_id, include_tool_discovery=False).tool_discovery is None
    )
    assert store.commit_run_input(command).session_revision == 1
    with pytest.raises(IrisRunPersistenceError):
        store.load_run_context(command.run_id, include_tool_discovery=True)


def test_memory_context_only_copies_returned_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧前缀增长不会增加局部读取和窄 header 的消息深拷贝数。"""
    store = InMemoryLifecycleStore()
    _seed(store, _history(2000), 1998)
    copied: list[Msg] = []
    original = Msg.__deepcopy__

    def copy_message(message: Msg, memo: dict[int, object]) -> Msg:
        copied.append(message)
        return original(message, memo)

    monkeypatch.setattr(Msg, "__deepcopy__", copy_message)
    assert store.load_session_header("session").message_count == 2000
    assert copied == []
    context = store.load_run_context("run-input", include_tool_discovery=True)
    assert len(copied) == len(context.raw_tail) + len(context.protected_prefix_messages) == 5


@pytest.mark.parametrize("expired", [False, True])
def test_memory_run_admission_does_not_copy_existing_history(
    monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    """普通准入与立即终止均复用内部历史，公开读取仍复制隔离。"""
    store = InMemoryLifecycleStore()
    messages = _history(2000)
    current = _seed(store, messages, 0)
    store.finish_run(
        FinishRun(
            run_id=current.run.run_id,
            expected_run_revision=current.run.revision,
            activation_id=current.run.current_activation_id,
            stop_reason=RunStopReason.COMPLETED,
            now=NOW,
        )
    )
    command = replace(
        _create_command(session_id="session", session_revision=1),
        options=AgentRunOptions(limits=RunLimits(deadline_at=NOW if expired else None)),
        now=NOW,
    )
    copied: list[Msg] = []
    original = Msg.__deepcopy__

    def copy_message(message: Msg, memo: dict[int, object]) -> Msg:
        copied.append(message)
        return original(message, memo)

    monkeypatch.setattr(Msg, "__deepcopy__", copy_message)
    created = store.create_run(command)
    assert created.run.phase is (RunPhase.TERMINAL if expired else RunPhase.ACTIVE)
    assert created.run.initial_session_message_count == 2000
    assert copied == []
    public = store.load_session("session")
    public.messages[0].metadata["changed"] = True
    assert store.load_session("session").messages == messages


def test_sqlite_unused_prefix_input_candidate_is_not_decoded(tmp_path: Path) -> None:
    """无 BCI 时，预取的第二条定位候选不进入原文解析或返回值。"""
    store = SQLiteStore(tmp_path / "unused.db")
    messages = _history(bci=False)
    _seed(store, messages, 18)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE session_messages SET message_json = 'not json' WHERE ordinal = 2"
        )
    snapshot = store.load_run_context("run-input", include_tool_discovery=True)
    assert snapshot.protected_prefix_messages == ((0, messages[0]), (17, messages[17]))
    assert snapshot.tool_discovery == _replay(messages).tool_discovery


def test_context_decode_releases_transaction_and_instance_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """解码已取得的后缀时，另一连接能提交且同实例能继续窄读取。"""
    store = SQLiteStore(tmp_path / "released.db")
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
        reading = pool.submit(store.load_run_context, command.run_id, include_tool_discovery=True)
        try:
            assert started.wait(2)
            assert pool.submit(store.load_session_header, "session").result(2).revision == 0
            writer = SQLiteStore(store.path)
            assert pool.submit(writer.commit_run_input, command).result(2).session_revision == 1
        finally:
            release.set()
        snapshot = reading.result(2)
    assert snapshot.header.revision == 0 and snapshot.raw_tail == ()
