"""所有 ``LifecycleStore`` 实现必须共享的 aggregate contract。"""

from __future__ import annotations

import sqlite3
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import pytest
from pydantic import ValidationError

from iris.exceptions import (
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    QuestionInteractionResponse,
    QuestionPrompt,
    SubagentExpiryOwner,
    SubagentProxyOrigin,
    ToolCallSnapshot,
)
from iris.lifecycle import (
    AdmitChildRun,
    AgentRunOptions,
    AgentRunRequest,
    CheckpointResumability,
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    CreateRun,
    FinalizeSubagentResult,
    FinishRun,
    LifecycleStore,
    RebindSubagentProxy,
    RecoverActiveRun,
    RecoveryDisposition,
    RequestCancellation,
    ReserveModelStep,
    ResolveInteraction,
    ResumeWaitingRun,
    RunCheckpoint,
    RunCommit,
    RunErrorInfo,
    RunLimits,
    RunRecord,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SessionSnapshot,
    SubagentRunLink,
    SuspendRun,
)
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolResult

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_T1 = _NOW + timedelta(seconds=1)
_T2 = _NOW + timedelta(seconds=2)
_T3 = _NOW + timedelta(seconds=3)
_ENVIRONMENT_FINGERPRINT = "environment-v1"
_TOOL_FINGERPRINT = "a" * 64


_INTERACTION_ID = "int_" + "1" * 32


class _StoreFactory(Protocol):
    def __call__(self) -> LifecycleStore: ...


@pytest.fixture(params=["memory", "sqlite"])
def lifecycle_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """让进程内与 SQLite 实现运行完全相同的 aggregate contract。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "lifecycle.db")
    return InMemoryLifecycleStore()


def _checkpoint(
    *,
    run_id: str,
    sequence: int,
    activation_id: str,
    session_revision: int,
    reserved: int = 0,
    committed: int = 0,
) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        sequence=sequence,
        activation_id=activation_id,
        engine_cursor={"step_index": committed},
        session_revision=session_revision,
        model_steps_reserved=reserved,
        model_steps_committed=committed,
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        resumability="safe",
    )


def _create_command(
    *,
    run_id: str = "run-1",
    session_id: str = "session-1",
    activation_id: str = "activation-1",
    max_model_steps: int = 20,
    metadata: dict[str, object] | None = None,
    session_revision: int = 0,
) -> CreateRun:
    return CreateRun(
        request=AgentRunRequest(
            input="start",
            session_id=session_id,
            run_id=run_id,
            metadata={} if metadata is None else metadata,
        ),
        options=AgentRunOptions(limits=RunLimits(max_model_steps=max_model_steps)),
        agent_id="agent-1",
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        start_activation_id=activation_id,
        initial_checkpoint=_checkpoint(
            run_id=run_id,
            sequence=1,
            activation_id=activation_id,
            session_revision=session_revision,
        ),
        now=_NOW,
    )


def _create(store: LifecycleStore, **kwargs: object) -> RunCommit:
    return store.create_run(_create_command(**kwargs))


def _interaction() -> HumanInteraction:
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


def _suspend(
    store: LifecycleStore,
    created: RunCommit,
    *,
    include_tool_history: bool = False,
) -> RunCommit:
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
    assistant = Msg.assistant(
        [ToolUseBlock(id="call-question", name="ask_question", input={"question": "继续吗？"})]
    )
    session_revision = 1 if include_tool_history else 0
    return store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant] if include_tool_history else [],
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=session_revision,
            ),
            pending_interaction=_interaction(),
            usage=created.run.usage,
            now=_T1,
        )
    )


def _resolve(store: LifecycleStore, waiting: RunCommit) -> tuple[ResolveInteraction, RunCommit]:
    command = ResolveInteraction(
        run_id="run-1",
        expected_run_revision=waiting.run.revision,
        interaction_id=_INTERACTION_ID,
        expected_interaction_version=waiting.interaction.version,
        response=QuestionInteractionResponse(answer="继续"),
        expected_fingerprint=_TOOL_FINGERPRINT,
        now=_T2,
    )
    return command, store.resolve_interaction(command)


def _prepare_tool(store: LifecycleStore) -> RunCommit:
    created = _create(store)
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
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
    return store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                run_id="run-1",
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


def _prepare_tool_batch(store: LifecycleStore) -> RunCommit:
    """为 claim 顺序测试持久化三条同 batch prepared call。"""
    created = _create(store)
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    uses = tuple(
        ToolUseBlock(id=f"call-{ordinal}", name="probe", input={"value": ordinal})
        for ordinal in range(1, 4)
    )
    assistant = Msg.assistant(uses)
    prepared = [
        RunToolCallRecord(
            run_id="run-1",
            step_index=0,
            ordinal=ordinal,
            tool_call_id=tool_use.id,
            tool_name=tool_use.name,
            arguments=dict(tool_use.input),
            fingerprint=_TOOL_FINGERPRINT,
            phase="prepared",
            version=1,
            created_at=_T1,
            updated_at=_T1,
        )
        for ordinal, tool_use in enumerate(uses, start=1)
    ]
    return store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=prepared,
            checkpoint=_checkpoint(
                run_id="run-1",
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


def test_terminal_cutoff_counts_messages_after_closure(
    lifecycle_store: LifecycleStore,
) -> None:
    """一批多个工具结果的截点按消息数记录，包含终态闭合消息。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    terminal = lifecycle_store.finish_run(
        FinishRun(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            stop_reason=RunStopReason.CANCELLED,
            now=_T2,
        )
    )
    session = lifecycle_store.load_session("session-1")
    assert len(session.messages) == 4
    assert terminal.run.terminal_session_message_count == 4
    assert session.revision == 2
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 4


def test_deadline_at_creation_has_zero_cutoff_without_checkpoint(
    lifecycle_store: LifecycleStore,
) -> None:
    """创建时已过期的 run 没有提交输入，仍保留零消息截点。"""
    expired = lifecycle_store.create_run(
        replace(
            _create_command(),
            options=AgentRunOptions(limits=RunLimits(deadline_at=_NOW)),
        )
    )
    assert expired.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    assert expired.run.terminal_session_message_count == 0
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 0
    assert lifecycle_store.load_checkpoint("run-1") is None


def test_deadline_at_creation_keeps_existing_history_cutoff(
    lifecycle_store: LifecycleStore,
) -> None:
    """创建时过期的后续 run 沿用已提交历史，不计入尚未提交的输入。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    lifecycle_store.finish_run(
        FinishRun(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            stop_reason=RunStopReason.CANCELLED,
            now=_T2,
        )
    )
    expired = lifecycle_store.create_run(
        replace(
            _create_command(run_id="run-2", activation_id="activation-2", session_revision=2),
            options=AgentRunOptions(limits=RunLimits(deadline_at=_NOW)),
        )
    )
    assert expired.run.terminal_session_message_count == 4
    assert lifecycle_store.load_run("run-2").terminal_session_message_count == 4
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 4
    assert len(lifecycle_store.load_session("session-1").messages) == 4
    assert lifecycle_store.load_checkpoint("run-2") is None


@pytest.mark.parametrize("include_tool_history", [False, True])
def test_waiting_cancellation_records_terminal_cutoff(
    lifecycle_store: LifecycleStore,
    include_tool_history: bool,
) -> None:
    """等待中取消将实际追加的工具闭合消息计入截点。"""
    waiting = _suspend(
        lifecycle_store,
        _create(lifecycle_store),
        include_tool_history=include_tool_history,
    )
    assert waiting.run.terminal_session_message_count is None
    terminal = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            reason="stop",
            settle_waiting=True,
            now=_T2,
        )
    )
    expected_count = 2 if include_tool_history else 1
    assert terminal.run.stop_reason is RunStopReason.CANCELLED
    assert terminal.run.terminal_session_message_count == expected_count
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == expected_count
    assert len(lifecycle_store.load_session("session-1").messages) == expected_count


def test_recovery_without_closer_keeps_committed_message_cutoff(
    lifecycle_store: LifecycleStore,
) -> None:
    """无工具闭合消息时，恢复结算仍记录已有历史的消息数。"""
    created = _create(lifecycle_store)
    reserved = lifecycle_store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    assistant = Msg.assistant("done")
    committed = lifecycle_store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[Msg.user("start"), assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=1,
                reserved=1,
                committed=1,
            ).model_copy(update={"resumability": CheckpointResumability.OUTCOME_READY}),
            assistant_message=assistant,
            now=_T1,
        )
    )
    command = RecoverActiveRun(
        run_id="run-1",
        expected_run_revision=committed.run.revision,
        expected_activation_id="activation-1",
        expected_checkpoint_sequence=committed.checkpoint.sequence,
        recovery_disposition=RecoveryDisposition.FINALIZE,
        now=_T2,
    )
    terminal = lifecycle_store.recover_active_run(command)
    assert terminal.run.phase.value == "terminal"
    assert terminal.run.terminal_session_message_count == 2
    assert terminal.session_revision is None
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 2
    assert lifecycle_store.load_session("session-1").revision == 1
    assert lifecycle_store.recover_active_run(command).run.terminal_session_message_count == 2
    if isinstance(lifecycle_store, SQLiteStore):
        reopened = SQLiteStore(lifecycle_store.path)
        assert reopened.load_run("run-1").terminal_session_message_count == 2


@pytest.mark.parametrize(
    ("phase", "count"),
    [("active", 0), ("waiting", 0), ("terminal", None), ("terminal", -1)],
)
def test_run_record_parsing_rejects_inconsistent_terminal_cutoff(
    phase: str,
    count: int | None,
) -> None:
    """原始记录解析时，消息截点必须与运行阶段一致且非负。"""
    raw = _create(InMemoryLifecycleStore()).run.model_dump(mode="json")
    raw.update(phase=phase, terminal_session_message_count=count)
    if phase != "active":
        raw["current_activation_id"] = None
    if phase == "waiting":
        raw["pending_interaction_id"] = _INTERACTION_ID
    elif phase == "terminal":
        raw.update(stop_reason="completed", finished_at=_T1.isoformat())

    with pytest.raises(ValidationError):
        RunRecord.model_validate(raw)


@pytest.mark.parametrize("messages", [[], [Msg.user("continue"), Msg.assistant("done")]])
def test_memory_session_append_preserves_direct_source(messages: list[Msg]) -> None:
    """追加历史和空增量都保留会话直接来源。"""
    original = SessionSnapshot(
        session_id="branch",
        messages=[Msg.user("inherited")],
        forked_from_run_id="source-run",
    )
    appended = InMemoryLifecycleStore._append_messages(original, messages)
    assert appended.forked_from_run_id == "source-run"
    assert appended.messages == [*original.messages, *messages]
    assert appended.revision == (1 if messages else 0)


def test_create_and_read_are_copy_isolated(lifecycle_store: LifecycleStore) -> None:
    """修改 command 或 read snapshot 不得改写 store 内部事实。"""
    command = _create_command(metadata={"nested": {"value": "original"}})
    lifecycle_store.create_run(command)
    command.request.metadata["nested"]["value"] = "changed"
    first = lifecycle_store.load_run("run-1")
    assert first is not None
    assert first.terminal_session_message_count is None
    first.request.metadata["nested"]["value"] = "also-changed"

    loaded = lifecycle_store.load_run("run-1")
    assert loaded is not None
    assert loaded.request.metadata == {"nested": {"value": "original"}}


def test_exact_tool_call_read_is_copy_isolated_and_missing_is_none(
    lifecycle_store: LifecycleStore,
) -> None:
    """Exact composite-key read 不扫描或模糊匹配其他 durable subject。"""
    _prepare_tool(lifecycle_store)
    _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-2",
        activation_id="activation-2",
    )

    loaded = lifecycle_store.load_tool_call("run-1", "call-tool")

    assert loaded == lifecycle_store.list_tool_calls("run-1")[0]
    assert loaded is not None
    loaded.arguments["value"] = "changed"
    reloaded = lifecycle_store.load_tool_call("run-1", "call-tool")
    assert reloaded is not None
    assert reloaded.arguments == {"value": "A"}
    assert lifecycle_store.load_tool_call("run-1", "missing") is None
    assert lifecycle_store.load_tool_call("missing", "call-tool") is None
    assert lifecycle_store.load_tool_call("run-2", "call-tool") is None
    with pytest.raises(IrisRunNotFoundError):
        lifecycle_store.list_tool_calls("missing")


def test_sqlite_corrupt_point_read_maps_decode_to_persistence_error(tmp_path: Path) -> None:
    """Exact tool read 保留既有 corrupt-row error context。"""
    store = SQLiteStore(tmp_path / "corrupt-tool.db")
    _prepare_tool(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE run_tool_calls SET arguments_json = '{' WHERE tool_call_id = 'call-tool'"
        )

    with pytest.raises(IrisRunPersistenceError) as captured:
        store.load_tool_call("run-1", "call-tool")

    assert captured.value.context["operation"] == "load_tool_call"
    assert captured.value.context["path"] == str(store.path)


def test_session_lane_read_tracks_non_terminal_owner_without_mutation(
    lifecycle_store: LifecycleStore,
) -> None:
    """Lane read 只观察 active/waiting owner，并在 terminal 后返回空。"""
    assert lifecycle_store.load_session_lane("session-1") is None
    created = _create(lifecycle_store)
    run_before = lifecycle_store.load_run("run-1")
    session_before = lifecycle_store.load_session("session-1")
    checkpoint_before = lifecycle_store.load_checkpoint("run-1")
    events_before = lifecycle_store.list_events("run-1")

    assert lifecycle_store.load_session_lane("session-1") == "run-1"
    assert lifecycle_store.load_session_lane("session-1") == "run-1"
    assert lifecycle_store.load_run("run-1") == run_before
    assert lifecycle_store.load_session("session-1") == session_before
    assert lifecycle_store.load_checkpoint("run-1") == checkpoint_before
    assert lifecycle_store.list_events("run-1") == events_before
    if isinstance(lifecycle_store, SQLiteStore):
        assert SQLiteStore(lifecycle_store.path).load_session_lane("session-1") == "run-1"

    waiting = _suspend(lifecycle_store, created)
    assert lifecycle_store.load_session_lane("session-1") == "run-1"
    if isinstance(lifecycle_store, SQLiteStore):
        assert SQLiteStore(lifecycle_store.path).load_session_lane("session-1") == "run-1"

    lifecycle_store.finish_run(
        FinishRun(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            activation_id=None,
            stop_reason=RunStopReason.CANCELLED,
            now=_T2,
        )
    )
    assert lifecycle_store.load_session_lane("session-1") is None
    if isinstance(lifecycle_store, SQLiteStore):
        assert SQLiteStore(lifecycle_store.path).load_session_lane("session-1") is None

    second = _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-1",
        activation_id="activation-2",
        session_revision=lifecycle_store.load_session("session-1").revision,
    )
    requested = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-2",
            expected_run_revision=second.run.revision,
            activation_id="activation-2",
            reason="stop",
            now=_T2,
        )
    )
    assert lifecycle_store.load_session_lane("session-1") == "run-2"
    lifecycle_store.finish_run(
        FinishRun(
            run_id="run-2",
            expected_run_revision=requested.run.revision,
            activation_id="activation-2",
            stop_reason=RunStopReason.CANCELLED,
            now=_T3,
        )
    )
    assert lifecycle_store.load_session_lane("session-1") is None


def test_reserve_exact_replay_is_noop_and_budget_exhaustion_is_terminal(
    lifecycle_store: LifecycleStore,
) -> None:
    """精确 replay 不增加 revision/event，下一 reservation 报预算耗尽。"""
    created = _create(lifecycle_store, max_model_steps=1)
    command = ReserveModelStep(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        now=_T1,
    )
    first = lifecycle_store.reserve_model_step(command)
    events_after_first = lifecycle_store.list_events("run-1")

    replay = lifecycle_store.reserve_model_step(command)
    assert replay.events == ()
    assert replay.run.revision == first.run.revision
    assert lifecycle_store.list_events("run-1") == events_after_first

    assistant = Msg.assistant("done")
    committed = lifecycle_store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=first.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[Msg.user("start"), assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            checkpoint=_checkpoint(
                run_id="run-1",
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
    terminal = lifecycle_store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=committed.run.revision,
            activation_id="activation-1",
            now=_T2,
        )
    )
    assert terminal.run.stop_reason == "budget_exhausted"
    assert terminal.result is not None
    assert terminal.run.terminal_session_message_count == 2
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 2


def test_commit_model_step_updates_history_checkpoint_and_tool_intents_atomically(
    lifecycle_store: LifecycleStore,
) -> None:
    """缺少任一 history/checkpoint/tool replacement 都应被本断言捕获。"""
    committed = _prepare_tool(lifecycle_store)

    session = lifecycle_store.load_session("session-1")
    assert session.revision == 1
    assert len(session.messages) == 1
    assert committed.checkpoint.sequence == 2
    assert lifecycle_store.list_tool_calls("run-1")[0].phase == "prepared"

    session.messages.clear()
    assert len(lifecycle_store.load_session("session-1").messages) == 1
    assert committed.session_revision == session.revision
    loaded = lifecycle_store.load_session("session-1")
    loaded.messages[0].metadata["caller-mutated"] = True
    assert lifecycle_store.load_session("session-1").messages[0].metadata == {}


def test_tool_batch_read_filters_before_returning_isolated_records(
    lifecycle_store: LifecycleStore,
) -> None:
    """按当前模型步限定工具记录，保留排序与返回值隔离。"""
    _prepare_tool_batch(lifecycle_store)
    assert lifecycle_store.list_tool_calls("run-1", step_index=1) == []
    current = lifecycle_store.list_tool_calls("run-1", step_index=0)
    assert [call.ordinal for call in current] == [1, 2, 3]
    current[0].arguments["value"] = "changed"
    assert lifecycle_store.load_tool_call("run-1", "call-1").arguments == {"value": 1}


def test_run_control_preserves_exact_session_identity(lifecycle_store: LifecycleStore) -> None:
    """窄控制快照携带所属 session，供 gateway 区分不同会话的 run。"""
    _create(lifecycle_store)
    _create(lifecycle_store, run_id="run-2", session_id="session-2", activation_id="activation-2")
    first = lifecycle_store.load_run_control("run-1")
    second = lifecycle_store.load_run_control("run-2")
    assert first.session_id == "session-1"
    assert second.session_id == "session-2"
    assert first.run_id == "run-1"
    assert second.run_id == "run-2"


def test_sqlite_mutation_receipt_does_not_read_session_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """提交 delta 只返回 revision；完整 history 由显式读取取得。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")

    def reject_history(*args: object, **kwargs: object) -> None:
        pytest.fail("mutation receipt must not load full session history")

    monkeypatch.setattr(store, "_select_session", reject_history)
    committed = _prepare_tool(store)
    assert committed.session_revision == 1


def test_sqlite_reads_do_not_deepcopy_newly_decoded_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite row 解析已经创建独立对象，读取无需再次复制。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _prepare_tool(store)

    def reject_copy(value: object) -> None:
        pytest.fail("SQLite read must return its newly decoded facts directly")

    monkeypatch.setattr("iris.store.sqlite.deepcopy", reject_copy)
    assert store.load_run("run-1") is not None
    assert len(store.load_session("session-1").messages) == 1
    assert store.load_checkpoint("run-1") is not None
    assert store.load_tool_call("run-1", "call-tool") is not None
    assert len(store.list_tool_calls("run-1")) == 1
    assert store.list_events("run-1")


def test_replay_retains_only_fact_identifiers(lifecycle_store: LifecycleStore) -> None:
    """精确重试缓存只保留重载事实所需的标识，不保存旧 aggregate。"""
    _prepare_tool(lifecycle_store)
    for replay in lifecycle_store._replays.values():
        assert all(
            isinstance(getattr(replay, item.name), str | bool | type(None))
            for item in fields(replay)
        )


def test_replay_key_is_encoded_once_per_mutation(
    lifecycle_store: LifecycleStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同次提交复用 canonical key；命中重试仍精确匹配完整命令。"""
    created = _create(lifecycle_store)
    module = import_module(type(lifecycle_store).__module__)
    original = module.replay_key
    encodings = 0

    def encode(operation: str, command: object) -> str:
        nonlocal encodings
        encodings += 1
        return original(operation, command)

    monkeypatch.setattr(module, "replay_key", encode)
    command = ReserveModelStep(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        now=_T1,
    )
    lifecycle_store.reserve_model_step(command)
    assert encodings == 1
    assert lifecycle_store.reserve_model_step(command).events == ()
    assert encodings == 2


@pytest.mark.parametrize(
    "usage_json",
    [
        '{"model_steps_reserved": -1}',
        '{"model_steps_reserved": 0, "model_steps_committed": 1}',
        '{"tool_calls_committed": -1}',
    ],
)
def test_sqlite_usage_json_is_validated_at_row_load(tmp_path: Path, usage_json: str) -> None:
    """单一 usage JSON 仍由既有 RunUsage owner 校验计数不变量。"""
    store = SQLiteStore(tmp_path / "lifecycle.db")
    _create(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE agent_runs SET usage_json = ?", (usage_json,))
    with pytest.raises(IrisRunPersistenceError):
        store.load_run("run-1")


def test_model_commit_replay_returns_current_facts_without_repeating_events(
    lifecycle_store: LifecycleStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """稍后重试旧提交会取得终态与最新 revision，且不重复消息和事件。"""
    commands: list[CommitModelStep] = []
    original = lifecycle_store.commit_model_step

    def capture(command: CommitModelStep) -> RunCommit:
        commands.append(command)
        return original(command)

    monkeypatch.setattr(lifecycle_store, "commit_model_step", capture)
    committed = _prepare_tool(lifecycle_store)
    terminal = lifecycle_store.finish_run(
        FinishRun(
            run_id="run-1",
            expected_run_revision=committed.run.revision,
            activation_id="activation-1",
            stop_reason=RunStopReason.CANCELLED,
            now=_T3,
        )
    )
    events = lifecycle_store.list_events("run-1")
    session = lifecycle_store.load_session("session-1")
    replay = original(commands[0])
    assert replay.run == terminal.run
    assert replay.result == terminal.result
    assert replay.checkpoint == terminal.checkpoint
    assert replay.session_revision == session.revision == 2
    assert replay.events == ()
    assert lifecycle_store.list_events("run-1") == events
    assert lifecycle_store.load_session("session-1") == session
    with pytest.raises(IrisRunConflictError):
        original(replace(commands[0], now=_T2))


def test_claim_and_commit_tool_result_cover_effect_fence(
    lifecycle_store: LifecycleStore,
) -> None:
    """工具 effect 必须存在 durable claim，result 提交推进 tool/session/checkpoint。"""
    prepared = _prepare_tool(lifecycle_store)
    claim = ClaimToolCall(
        run_id="run-1",
        expected_run_revision=prepared.run.revision,
        activation_id="activation-1",
        tool_call_id="call-tool",
        fingerprint=_TOOL_FINGERPRINT,
        expected_tool_version=1,
        now=_T2,
    )
    claimed = lifecycle_store.claim_tool_call(claim)
    assert lifecycle_store.claim_tool_call(claim).events == ()
    claimed_call = lifecycle_store.list_tool_calls("run-1")[0]
    result = ToolResult(
        tool_use_id="call-tool",
        tool_name="probe",
        content=[TextBlock(text="done")],
    )

    committed = lifecycle_store.commit_tool_result(
        CommitToolResult(
            run_id="run-1",
            expected_run_revision=claimed.run.revision,
            activation_id="activation-1",
            expected_session_revision=1,
            tool_call_id="call-tool",
            expected_tool_version=claimed_call.version,
            result=result,
            message_delta=[
                Msg.tool_result(
                    tool_use_id="call-tool",
                    name="probe",
                    content="done",
                )
            ],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=3,
                activation_id="activation-1",
                session_revision=2,
                reserved=1,
                committed=1,
            ),
            now=_T3,
        )
    )
    assert committed.run.usage.tool_calls_committed == 1
    assert lifecycle_store.list_tool_calls("run-1")[0].result == result
    assert lifecycle_store.load_session("session-1").revision == 2


def test_claim_batch_respects_durable_cancellation_fence(
    lifecycle_store: LifecycleStore,
) -> None:
    """取消前可多 claim；取消后只允许 exact replay 与既有 claim result。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    first_command = ClaimToolCall(
        run_id="run-1",
        expected_run_revision=prepared.run.revision,
        activation_id="activation-1",
        tool_call_id="call-1",
        fingerprint=_TOOL_FINGERPRINT,
        expected_tool_version=1,
        now=_T2,
    )
    first = lifecycle_store.claim_tool_call(first_command)
    second = lifecycle_store.claim_tool_call(
        replace(
            first_command,
            expected_run_revision=first.run.revision,
            tool_call_id="call-2",
        )
    )
    cancelled = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-1",
            expected_run_revision=second.run.revision,
            activation_id="activation-1",
            reason="user requested",
            now=_T3,
        )
    )
    events_after_cancel = lifecycle_store.list_events("run-1")

    replay = lifecycle_store.claim_tool_call(first_command)
    assert replay.events == ()
    with pytest.raises(IrisRunStateError, match="取消"):
        lifecycle_store.claim_tool_call(
            replace(
                first_command,
                expected_run_revision=cancelled.run.revision,
                tool_call_id="call-3",
            )
        )
    assert lifecycle_store.list_events("run-1") == events_after_cancel
    calls = {call.tool_call_id: call for call in lifecycle_store.list_tool_calls("run-1")}
    assert calls["call-1"].phase == "claimed"
    assert calls["call-2"].phase == "claimed"
    assert calls["call-3"].phase == "prepared"

    committed = lifecycle_store.commit_tool_result(
        CommitToolResult(
            run_id="run-1",
            expected_run_revision=cancelled.run.revision,
            activation_id="activation-1",
            expected_session_revision=1,
            tool_call_id="call-1",
            expected_tool_version=2,
            result=ToolResult(tool_use_id="call-1", tool_name="probe"),
            message_delta=[Msg.tool_result(tool_use_id="call-1", name="probe")],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=3,
                activation_id="activation-1",
                session_revision=2,
                reserved=1,
                committed=1,
            ),
            now=_T3,
        )
    )
    assert committed.run.usage.tool_calls_committed == 1


@pytest.mark.parametrize(
    ("tool_call_id", "fingerprint", "error_type"),
    [
        ("missing-call", _TOOL_FINGERPRINT, IrisRunNotFoundError),
        ("call-1", "b" * 64, IrisRunConflictError),
    ],
)
def test_cancelled_claim_preserves_exact_subject_error_priority(
    lifecycle_store: LifecycleStore,
    tool_call_id: str,
    fingerprint: str,
    error_type: type[Exception],
) -> None:
    """Cancellation fence 不得遮蔽不存在或 fingerprint 错误。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    cancelled = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            reason="user requested",
            now=_T2,
        )
    )
    events_after_cancel = lifecycle_store.list_events("run-1")

    with pytest.raises(error_type):
        lifecycle_store.claim_tool_call(
            ClaimToolCall(
                run_id="run-1",
                expected_run_revision=cancelled.run.revision,
                activation_id="activation-1",
                tool_call_id=tool_call_id,
                fingerprint=fingerprint,
                expected_tool_version=1,
                now=_T3,
            )
        )

    assert lifecycle_store.list_events("run-1") == events_after_cancel
    assert all(call.phase == "prepared" for call in lifecycle_store.list_tool_calls("run-1"))


def test_resolve_exact_response_replays_but_different_response_conflicts(
    lifecycle_store: LifecycleStore,
) -> None:
    """同 response 幂等，改变 response 不能覆盖已经 durable 的人工事实。"""
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    command, resolved = _resolve(lifecycle_store, waiting)
    replay = lifecycle_store.resolve_interaction(command)

    assert replay.events == ()
    assert replay.run.revision == resolved.run.revision
    refreshed_replay = lifecycle_store.resolve_interaction(
        replace(
            command,
            expected_run_revision=resolved.run.revision,
            expected_interaction_version=resolved.interaction.version,
        )
    )
    assert refreshed_replay.events == ()
    assert refreshed_replay.run == resolved.run
    with pytest.raises(IrisRunConflictError):
        lifecycle_store.resolve_interaction(
            replace(command, response=QuestionInteractionResponse(answer="停止"))
        )


def test_resume_waiting_run_rebinds_checkpoint_and_clears_waiting_result(
    lifecycle_store: LifecycleStore,
) -> None:
    """Resume 创建新 fence，并关闭旧 interaction 与 active result。"""
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    _, resolved = _resolve(lifecycle_store, waiting)
    resumed = lifecycle_store.resume_waiting_run(
        ResumeWaitingRun(
            run_id="run-1",
            expected_run_revision=resolved.run.revision,
            new_activation_id="activation-2",
            kind="resume",
            expected_checkpoint_sequence=resolved.checkpoint.sequence,
            now=_T3,
        )
    )

    assert resumed.checkpoint.sequence == resolved.checkpoint.sequence + 1
    assert resumed.interaction.status == "closed"
    assert lifecycle_store.load_result("run-1") is None


def test_suspend_rejects_interaction_subject_that_differs_from_prepared_call(
    lifecycle_store: LifecycleStore,
) -> None:
    created = _create(lifecycle_store)
    interaction = _interaction()
    mismatched_subject = interaction.request.tool_call.model_copy(
        update={"arguments": {"question": "另一个问题"}}
    )
    interaction = interaction.model_copy(
        update={"request": interaction.request.model_copy(update={"tool_call": mismatched_subject})}
    )
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

    with pytest.raises(IrisRunConflictError, match="subject"):
        lifecycle_store.suspend_run(
            SuspendRun(
                run_id="run-1",
                expected_run_revision=created.run.revision,
                activation_id="activation-1",
                expected_session_revision=0,
                prepared_tool_calls=[prepared],
                checkpoint=_checkpoint(
                    run_id="run-1",
                    sequence=2,
                    activation_id="activation-1",
                    session_revision=0,
                ),
                pending_interaction=interaction,
                usage=created.run.usage,
                now=_T1,
            )
        )


def test_question_projection_must_match_exact_durable_answer(
    lifecycle_store: LifecycleStore,
) -> None:
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    _, resolved = _resolve(lifecycle_store, waiting)
    begun = lifecycle_store.resume_waiting_run(
        ResumeWaitingRun(
            run_id="run-1",
            expected_run_revision=resolved.run.revision,
            new_activation_id="activation-resume",
            kind="resume",
            expected_checkpoint_sequence=2,
            now=_T3,
        )
    )
    assert begun.checkpoint is not None

    with pytest.raises(IrisRunStateError, match="claim"):
        lifecycle_store.commit_tool_result(
            CommitToolResult(
                run_id="run-1",
                expected_run_revision=begun.run.revision,
                activation_id="activation-resume",
                expected_session_revision=0,
                tool_call_id="call-question",
                expected_tool_version=1,
                result=ToolResult(
                    tool_use_id="call-question",
                    tool_name="ask_question",
                    content=[TextBlock(text="伪造答案")],
                    data={"answer": "伪造答案"},
                ),
                checkpoint=_checkpoint(
                    run_id="run-1",
                    sequence=begun.checkpoint.sequence + 1,
                    activation_id="activation-resume",
                    session_revision=0,
                ),
                now=_T3,
            )
        )


def test_cancellation_request_is_once_only_and_does_not_release_active_lane(
    lifecycle_store: LifecycleStore,
) -> None:
    """Active cancellation 只记录 durable intent，不冒充已结算 terminal。"""
    created = _create(lifecycle_store)
    command = RequestCancellation(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        reason="user requested",
        now=_T1,
    )
    requested = lifecycle_store.request_cancellation(command)
    event_count = len(lifecycle_store.list_events("run-1"))

    assert lifecycle_store.request_cancellation(command).events == ()
    refreshed_replay = lifecycle_store.request_cancellation(
        replace(command, expected_run_revision=requested.run.revision)
    )
    assert refreshed_replay.events == ()
    assert refreshed_replay.run == requested.run
    assert len(lifecycle_store.list_events("run-1")) == event_count
    assert requested.run.phase == "active"
    assert requested.run.terminal_session_message_count is None
    with pytest.raises(IrisRunConflictError):
        _create(
            lifecycle_store,
            run_id="run-2",
            session_id="session-1",
            activation_id="activation-2",
        )


def test_finish_exact_replay_is_noop_and_releases_lane(
    lifecycle_store: LifecycleStore,
) -> None:
    """Terminal event 只能出现一次，精确 finish replay 不制造新事实。"""
    created = _create(lifecycle_store)
    command = FinishRun(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        stop_reason=RunStopReason.COMPLETED,
        assistant_message=Msg.assistant("done"),
        now=_T1,
    )
    terminal = lifecycle_store.finish_run(command)
    replay = lifecycle_store.finish_run(command)

    assert replay.events == ()
    assert replay.run == terminal.run
    assert terminal.run.terminal_session_message_count == 0
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 0
    next_run = _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-1",
        activation_id="activation-2",
    )
    assert next_run.run.phase == "active"


def test_terminal_finish_closes_claimed_and_prepared_history_atomically(
    lifecycle_store: LifecycleStore,
) -> None:
    """Terminal settlement 必须原子闭合全部 unresolved tool history。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    claim_revision = prepared.run.revision
    for tool_call_id in ("call-1", "call-2"):
        claimed = lifecycle_store.claim_tool_call(
            ClaimToolCall(
                run_id="run-1",
                expected_run_revision=claim_revision,
                activation_id="activation-1",
                tool_call_id=tool_call_id,
                fingerprint=_TOOL_FINGERPRINT,
                expected_tool_version=1,
                now=_T2,
            )
        )
        claim_revision = claimed.run.revision

    command = FinishRun(
        run_id="run-1",
        expected_run_revision=claim_revision,
        activation_id="activation-1",
        stop_reason=RunStopReason.OUTCOME_UNKNOWN,
        error=RunErrorInfo(
            code="TOOL_OUTCOME_UNKNOWN",
            message="工具结果不可证明",
            source="tool",
        ),
        now=_T3,
    )
    terminal = lifecycle_store.finish_run(command)

    records = lifecycle_store.list_tool_calls("run-1")
    assert [record.phase for record in records] == [
        "outcome_unknown",
        "outcome_unknown",
        "prepared",
    ]
    assert [event.kind for event in terminal.events] == [
        "tool_call.outcome_unknown",
        "tool_call.outcome_unknown",
        "run.terminal",
    ]
    assert terminal.events[-1].kind == "run.terminal"
    assert {
        event.correlation_id
        for event in terminal.events
        if event.kind == "tool_call.outcome_unknown"
    } == {"call-1", "call-2"}
    assert all(record.result is None for record in records)

    session = lifecycle_store.load_session("session-1")
    tool_results = [result for message in session.messages for result in message.tool_results]
    assert session.revision == 2
    assert [result.tool_use_id for result in tool_results] == ["call-1", "call-2", "call-3"]
    assert [result.is_error for result in tool_results] == [True, True, True]
    assert [result.metadata["error"]["code"] for result in tool_results] == [
        "TOOL_OUTCOME_UNKNOWN",
        "TOOL_OUTCOME_UNKNOWN",
        "TOOL_NOT_STARTED",
    ]
    assert [result.metadata["error"]["retryable"] for result in tool_results] == [
        False,
        False,
        True,
    ]
    assert terminal.session_revision == session.revision
    assert terminal.checkpoint is not None
    assert terminal.checkpoint.sequence == 2
    assert terminal.checkpoint.session_revision == session.revision
    assert terminal.run.usage.tool_calls_committed == 0
    assert terminal.run.terminal_session_message_count == 4
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 4

    replay = lifecycle_store.finish_run(command)
    assert replay.events == ()
    assert replay.run.terminal_session_message_count == 4
    assert lifecycle_store.load_session("session-1") == session


def test_safe_recovery_abandons_old_fence_and_rebinds_checkpoint(
    lifecycle_store: LifecycleStore,
) -> None:
    """无未知 effect 时 recovery 必须创建全新 activation fence。"""
    created = _create(lifecycle_store)
    recovered = lifecycle_store.recover_active_run(
        RecoverActiveRun(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            expected_activation_id="activation-1",
            expected_checkpoint_sequence=1,
            recovery_disposition=RecoveryDisposition.RESUME,
            new_activation_id="activation-recovery",
            now=_T1,
        )
    )
    assert recovered.run.current_activation_id == "activation-recovery"
    assert recovered.checkpoint.sequence == 2
    assert [event.kind for event in recovered.events] == [
        "activation.abandoned",
        "activation.started",
    ]


def test_safe_recovery_rejects_unresolved_durable_claim(
    lifecycle_store: LifecycleStore,
) -> None:
    """Claimed effect 无 durable result 时只能 outcome_unknown，不能安全重放。"""
    prepared = _prepare_tool(lifecycle_store)
    claimed = lifecycle_store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )

    with pytest.raises(IrisRunRecoveryError):
        lifecycle_store.recover_active_run(
            RecoverActiveRun(
                run_id="run-1",
                expected_run_revision=claimed.run.revision,
                expected_activation_id="activation-1",
                expected_checkpoint_sequence=claimed.checkpoint.sequence,
                recovery_disposition=RecoveryDisposition.RESUME,
                new_activation_id="activation-recovery",
                now=_T3,
            )
        )


def test_outcome_unknown_recovery_roundtrips_exact_activation_and_tool_facts(
    lifecycle_store: LifecycleStore,
) -> None:
    """Recovery 原子关闭多个 claim，重开后保留 activation/tool 精确事实。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    first_claimed = lifecycle_store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            tool_call_id="call-1",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )
    claimed = lifecycle_store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=first_claimed.run.revision,
            activation_id="activation-1",
            tool_call_id="call-2",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )

    recovered = lifecycle_store.recover_active_run(
        RecoverActiveRun(
            run_id="run-1",
            expected_run_revision=claimed.run.revision,
            expected_activation_id="activation-1",
            expected_checkpoint_sequence=claimed.checkpoint.sequence,
            recovery_disposition=RecoveryDisposition.OUTCOME_UNKNOWN,
            now=_T3,
        )
    )

    assert [event.kind for event in recovered.events] == [
        "activation.abandoned",
        "tool_call.outcome_unknown",
        "tool_call.outcome_unknown",
        "run.terminal",
    ]
    records = lifecycle_store.list_tool_calls("run-1")
    assert [record.phase for record in records] == [
        "outcome_unknown",
        "outcome_unknown",
        "prepared",
    ]
    session = lifecycle_store.load_session("session-1")
    tool_results = [result for message in session.messages for result in message.tool_results]
    assert session.revision == 2
    assert [result.tool_use_id for result in tool_results] == ["call-1", "call-2", "call-3"]
    assert recovered.session_revision == session.revision
    assert recovered.checkpoint is not None
    assert recovered.checkpoint.sequence == claimed.checkpoint.sequence
    assert recovered.checkpoint.session_revision == session.revision
    assert recovered.run.terminal_session_message_count == 4
    assert lifecycle_store.load_run("run-1").terminal_session_message_count == 4
    if not isinstance(lifecycle_store, SQLiteStore):
        return
    reopened = SQLiteStore(lifecycle_store.path)
    assert reopened.load_run("run-1").terminal_session_message_count == 4
    reopened_records = reopened.list_tool_calls("run-1")
    with sqlite3.connect(lifecycle_store.path) as connection:
        activation_fact = connection.execute(
            "SELECT status, outcome FROM run_activations WHERE activation_id = ?",
            ("activation-1",),
        ).fetchone()

    assert activation_fact == ("abandoned", "outcome_unknown")
    assert [record.phase for record in reopened_records] == [
        "outcome_unknown",
        "outcome_unknown",
        "prepared",
    ]
    assert [record.updated_at for record in reopened_records[:2]] == [_T3, _T3]


def test_read_methods_apply_cursor_and_validation_contract(
    lifecycle_store: LifecycleStore,
) -> None:
    """Read API 不重排/修改 durable truth，且负游标 fail closed。"""
    _create(lifecycle_store)
    assert lifecycle_store.load_session("missing").revision == 0
    assert lifecycle_store.load_checkpoint("run-1").sequence == 1
    assert lifecycle_store.load_interaction("missing") is None
    assert lifecycle_store.load_result("run-1") is None
    assert [event.sequence for event in lifecycle_store.list_events("run-1", limit=1)] == [1]
    assert [event.sequence for event in lifecycle_store.list_events("run-1", 1, limit=1)] == [2]
    assert [event.sequence for event in lifecycle_store.list_events("run-1", 1)] == [2]
    with pytest.raises(IrisRunStateError):
        lifecycle_store.list_events("run-1", -1)
    for invalid_limit in (0, -1, True, 1.5):
        with pytest.raises(IrisRunStateError):
            lifecycle_store.list_events("run-1", limit=cast(int, invalid_limit))


def _subagent_parent(store: LifecycleStore) -> RunCommit:
    """用普通 model commit 构造 parent 的 prepared subagent 调用。"""
    created = _create(store, run_id="parent", session_id="parent-session", activation_id="parent-a")
    created = store.reserve_model_step(
        ReserveModelStep(
            run_id="parent",
            expected_run_revision=created.run.revision,
            activation_id="parent-a",
            now=_NOW,
        )
    )
    assistant = Msg.assistant(
        [ToolUseBlock(id="delegate", name="subagent", input={"prompt": "work"})]
    )
    return store.commit_model_step(
        CommitModelStep(
            run_id="parent",
            expected_run_revision=created.run.revision,
            activation_id="parent-a",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=[
                RunToolCallRecord(
                    run_id="parent",
                    step_index=0,
                    ordinal=1,
                    tool_call_id="delegate",
                    tool_name="subagent",
                    arguments={"prompt": "work"},
                    fingerprint=_TOOL_FINGERPRINT,
                    phase="prepared",
                    version=1,
                    created_at=_NOW,
                    updated_at=_NOW,
                )
            ],
            checkpoint=_checkpoint(
                run_id="parent",
                sequence=2,
                activation_id="parent-a",
                session_revision=1,
                reserved=1,
                committed=1,
            ),
            assistant_message=assistant,
            now=_NOW,
        )
    )


def _admit_child(store: LifecycleStore, parent: RunCommit) -> AdmitChildRun:
    command = AdmitChildRun(
        parent_run_id="parent",
        expected_parent_run_revision=parent.run.revision,
        parent_activation_id="parent-a",
        parent_tool_call_id="delegate",
        expected_parent_tool_version=1,
        child_create=_create_command(),
    )
    store.admit_child_run(command)
    return command


def _child_commit(store: LifecycleStore) -> RunCommit:
    return RunCommit(run=store.load_run("run-1"), checkpoint=store.load_checkpoint("run-1"))


def _proxy(
    child: HumanInteraction,
    *,
    interaction_id: str = "proxy-1",
    expires_at: datetime | None = None,
    expiry_owner: SubagentExpiryOwner | None = None,
) -> HumanInteraction:
    return HumanInteraction(
        interaction_id=interaction_id,
        session_id="parent-session",
        run_id="parent",
        step_index=0,
        tool_call_id="delegate",
        created_at=_NOW,
        expires_at=expires_at,
        request=HumanInteractionRequest(
            tool_call=ToolCallSnapshot(
                tool_call_id="delegate",
                tool_name="subagent",
                arguments={"prompt": "work"},
                workspace_root="workspace",
                fingerprint=_TOOL_FINGERPRINT,
            ),
            prompt=child.request.prompt,
            subagent_origin=SubagentProxyOrigin(
                child_run_id="run-1",
                child_interaction_id=child.interaction_id,
                agent_selector="researcher",
                expiry_owner=expiry_owner,
            ),
        ),
    )


def _bind_proxy(
    store: LifecycleStore,
    parent: RunCommit,
    proxy: HumanInteraction,
    *,
    replaced: str | None = None,
) -> RunCommit:
    return store.rebind_subagent_proxy(
        RebindSubagentProxy(
            parent_run_id="parent",
            expected_parent_run_revision=parent.run.revision,
            parent_activation_id=parent.run.current_activation_id,
            parent_tool_call_id="delegate",
            expected_parent_tool_version=1,
            pending_proxy=proxy,
            replaced_proxy_interaction_id=replaced,
            now=_T1,
        )
    )


def _resolve_proxy(store: LifecycleStore, waiting: RunCommit) -> RunCommit:
    return store.resolve_interaction(
        ResolveInteraction(
            run_id="parent",
            expected_run_revision=waiting.run.revision,
            interaction_id=waiting.interaction.interaction_id,
            expected_interaction_version=waiting.interaction.version,
            response=QuestionInteractionResponse(answer="continue"),
            expected_fingerprint=_TOOL_FINGERPRINT,
            now=_T2,
        )
    )


def _terminal_child(store: LifecycleStore) -> None:
    child = store.load_run("run-1")
    store.finish_run(
        FinishRun(
            run_id=child.run_id,
            expected_run_revision=child.revision,
            activation_id=child.current_activation_id,
            stop_reason=RunStopReason.COMPLETED,
            assistant_message=Msg.assistant("child answer"),
            now=_T3,
        )
    )


def _finalize_command(store: LifecycleStore, *, waiting: bool) -> FinalizeSubagentResult:
    parent = store.load_run("parent")
    checkpoint = store.load_checkpoint("parent")
    result = ToolResult(
        tool_use_id="delegate",
        tool_name="subagent",
        content=[TextBlock(text="done")],
        metadata={"agent_selector": "researcher", "child_run_id": "run-1"},
    )
    return FinalizeSubagentResult(
        parent_run_id="parent",
        expected_parent_run_revision=parent.revision,
        parent_activation_id=parent.current_activation_id,
        expected_parent_session_revision=1,
        parent_tool_call_id="delegate",
        expected_parent_tool_version=1,
        proxy_interaction_id=parent.pending_interaction_id,
        resume_activation_id="parent-resume" if waiting else None,
        result=result,
        message_delta=[result.to_msg()],
        checkpoint=checkpoint.model_copy(
            update={
                "sequence": checkpoint.sequence + 1,
                "activation_id": "parent-resume" if waiting else "parent-a",
                "session_revision": 2,
                "engine_cursor": {"next_tool_index": 1},
            }
        ),
        now=_T3,
    )


def test_admit_child_run_is_atomic_and_parent_stays_prepared(
    lifecycle_store: LifecycleStore,
) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _admit_child(store, parent)
    link = store.load_subagent_link("parent", "delegate")
    assert link == SubagentRunLink(
        parent_run_id="parent", parent_tool_call_id="delegate", child_run_id="run-1"
    )
    assert link.model_dump() == {
        "parent_run_id": "parent",
        "parent_tool_call_id": "delegate",
        "child_run_id": "run-1",
    }
    assert store.load_run("parent") == parent.run
    assert store.load_tool_call("parent", "delegate").phase.value == "prepared"
    assert store.load_run("run-1").phase.value == "active"
    assert store.load_session_lane("session-1") == "run-1"
    assert store.load_checkpoint("run-1").activation_id == "activation-1"
    assert len(store.list_events("run-1")) == 2
    if isinstance(store, SQLiteStore):
        reopened = SQLiteStore(store.path)
        assert reopened.load_subagent_link("parent", "delegate") == link
        assert reopened.load_run("run-1") == store.load_run("run-1")


def test_admit_child_run_reentry_returns_original_link_without_new_child(
    lifecycle_store: LifecycleStore,
) -> None:
    store = lifecycle_store
    command = _admit_child(store, _subagent_parent(store))
    replacement = replace(
        command,
        child_create=_create_command(
            run_id="other",
            session_id="other-session",
            activation_id="other-a",
        ),
    )
    assert store.admit_child_run(replacement).child_run_id == "run-1"
    assert store.load_run("other") is None
    assert len(store.list_events("run-1")) == 2


def test_admit_child_run_failure_leaves_no_child_or_link(lifecycle_store: LifecycleStore) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _create(store, run_id="occupied", activation_id="occupied-a")
    command = AdmitChildRun(
        parent_run_id="parent",
        expected_parent_run_revision=parent.run.revision,
        parent_activation_id="parent-a",
        parent_tool_call_id="delegate",
        expected_parent_tool_version=1,
        child_create=_create_command(),
    )
    with pytest.raises(IrisRunConflictError):
        store.admit_child_run(command)
    assert store.load_subagent_link("parent", "delegate") is None
    assert store.load_run("run-1") is None
    assert store.load_run("parent") == parent.run


def test_rebind_subagent_proxy_moves_active_parent_to_waiting_without_committing_tool(
    lifecycle_store: LifecycleStore,
) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _admit_child(store, parent)
    child = _suspend(store, _child_commit(store))
    proxy = _proxy(child.interaction)
    result = _bind_proxy(store, parent, proxy)
    assert result.run.phase.value == "waiting"
    assert result.run.current_activation_id is None
    assert result.run.pending_interaction_id == proxy.interaction_id
    assert result.result.pending_interaction == proxy
    assert result.interaction == store.load_interaction(proxy.interaction_id) == proxy
    assert result.checkpoint.engine_cursor == parent.checkpoint.engine_cursor
    assert result.checkpoint.sequence == parent.checkpoint.sequence + 1
    assert result.checkpoint.activation_id == "parent-a"
    assert result.run.usage == parent.run.usage
    call = store.load_tool_call("parent", "delegate")
    assert (call.phase.value, call.version, call.interaction_id) == ("prepared", 1, "proxy-1")
    assert store.load_session("parent-session").revision == 1
    assert [event.kind.value for event in result.events] == ["interaction.suspended"]


def test_rebind_subagent_proxy_rejects_mismatched_child_interaction_atomically(
    lifecycle_store: LifecycleStore,
) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _admit_child(store, parent)
    child = _suspend(store, _child_commit(store))
    proxy = _proxy(child.interaction.model_copy(update={"interaction_id": "wrong-child-question"}))
    with pytest.raises(IrisRunConflictError):
        _bind_proxy(store, parent, proxy)
    assert store.load_run("parent") == parent.run
    assert store.load_checkpoint("parent") == parent.checkpoint
    assert store.load_interaction("proxy-1") is None


def test_rebind_subagent_proxy_replaces_resolved_proxy_while_parent_stays_waiting(
    lifecycle_store: LifecycleStore,
) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _admit_child(store, parent)
    child = _suspend(store, _child_commit(store))
    waiting = _bind_proxy(store, parent, _proxy(child.interaction))
    resolved = _resolve_proxy(store, waiting)
    # 正常 child resume 后提交原问题回答，再进入第二个问题。
    _, child_resolved = _resolve(store, child)
    active = store.resume_waiting_run(
        ResumeWaitingRun(
            run_id="run-1",
            expected_run_revision=child_resolved.run.revision,
            new_activation_id="child-resume",
            kind="resume",
            expected_checkpoint_sequence=child_resolved.checkpoint.sequence,
            now=_T2,
        )
    )
    answer = ToolResult(
        tool_use_id="call-question",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
        data={"answer": "继续"},
    )
    answered = store.commit_tool_result(
        CommitToolResult(
            run_id="run-1",
            expected_run_revision=active.run.revision,
            activation_id="child-resume",
            expected_session_revision=0,
            tool_call_id="call-question",
            expected_tool_version=1,
            result=answer,
            message_delta=[answer.to_msg()],
            checkpoint=active.checkpoint.model_copy(
                update={"sequence": active.checkpoint.sequence + 1, "session_revision": 1}
            ),
            now=_T2,
        )
    )
    question = child.interaction.model_copy(
        update={
            "interaction_id": "child-question-2",
            "tool_call_id": "question-2",
            "request": child.interaction.request.model_copy(
                update={
                    "tool_call": child.interaction.request.tool_call.model_copy(
                        update={"tool_call_id": "question-2"}
                    ),
                }
            ),
        }
    )
    next_child = store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=answered.run.revision,
            activation_id="child-resume",
            expected_session_revision=1,
            checkpoint=answered.checkpoint.model_copy(
                update={"sequence": answered.checkpoint.sequence + 1}
            ),
            pending_interaction=question,
            usage=answered.run.usage,
            prepared_tool_calls=[
                RunToolCallRecord(
                    run_id="run-1",
                    step_index=0,
                    ordinal=2,
                    tool_call_id="question-2",
                    tool_name="ask_question",
                    arguments={"question": "继续吗？"},
                    fingerprint=_TOOL_FINGERPRINT,
                    phase="prepared",
                    version=1,
                    created_at=_T2,
                    updated_at=_T2,
                )
            ],
            now=_T2,
        )
    )
    rebound = _bind_proxy(
        store,
        resolved,
        _proxy(next_child.interaction, interaction_id="proxy-2"),
        replaced="proxy-1",
    )
    assert rebound.run.phase.value == "waiting"
    assert rebound.run.current_activation_id is None
    assert rebound.result.pending_interaction.interaction_id == "proxy-2"
    assert rebound.checkpoint.engine_cursor == waiting.checkpoint.engine_cursor
    assert rebound.checkpoint.activation_id == waiting.checkpoint.activation_id
    assert rebound.run.usage == parent.run.usage
    assert store.load_interaction("proxy-1").status == InteractionStatus.CLOSED
    assert (
        store.load_interaction("proxy-2").request.subagent_origin.child_interaction_id
        == "child-question-2"
    )
    assert store.load_session("parent-session").revision == 1


@pytest.mark.parametrize("mode", ["active", "resolved", "expired"])
def test_finalize_subagent_result_commits_parent_once_without_claim(
    lifecycle_store: LifecycleStore,
    mode: str,
) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _admit_child(store, parent)
    if mode != "active":
        child = _suspend(store, _child_commit(store))
        waiting = _bind_proxy(
            store,
            parent,
            _proxy(
                child.interaction,
                expires_at=_T3 if mode == "expired" else None,
                expiry_owner=SubagentExpiryOwner.OUTER_TOOL_TIMEOUT if mode == "expired" else None,
            ),
        )
        if mode == "resolved":
            _resolve_proxy(store, waiting)
    _terminal_child(store)
    command = _finalize_command(store, waiting=mode != "active")
    result = store.finalize_subagent_result(command)
    assert result.run.phase.value == "active"
    assert result.run.pending_interaction_id is None
    assert result.result is None
    assert (
        result.checkpoint.activation_id
        == result.run.current_activation_id
        == ("parent-a" if mode == "active" else "parent-resume")
    )
    assert result.checkpoint.engine_cursor == {"next_tool_index": 1}
    assert result.run.usage.tool_calls_committed == 1
    call = store.load_tool_call("parent", "delegate")
    assert call.phase.value == "committed"
    assert call.claim_activation_id is None
    assert call.result.model_content == "done"
    events = [event.kind.value for event in result.events]
    assert events == (
        ["tool_call.committed"]
        if mode == "active"
        else ["activation.started", "tool_call.committed"]
    )
    assert len(store.load_session("parent-session").messages) == 2
    if mode != "active":
        closed = store.load_interaction("proxy-1")
        assert closed.status == InteractionStatus.CLOSED
        if mode == "expired":
            assert closed.response is None
    replay = store.finalize_subagent_result(command)
    assert replay.run == result.run
    assert replay.events == ()
    assert store.load_run("parent").usage.tool_calls_committed == 1
    assert len(store.load_session("parent-session").messages) == 2


def test_finalize_subagent_result_rejects_nonterminal_child_without_parent_delta(
    lifecycle_store: LifecycleStore,
) -> None:
    store = lifecycle_store
    parent = _subagent_parent(store)
    _admit_child(store, parent)
    with pytest.raises(IrisRunStateError):
        store.finalize_subagent_result(_finalize_command(store, waiting=False))
    assert store.load_run("parent") == parent.run
    assert store.load_session("parent-session").revision == 1
