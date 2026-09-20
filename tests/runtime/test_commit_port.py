from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.harness._commit_port import StoreRuntimeCommitPort
from iris.harness._events import _RunEventCollector
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    RequestCancellation,
    RunCheckpoint,
    RunCommit,
    RunControlSnapshot,
    RunEvent,
    RunPhase,
    RunRecord,
    RunToolCallRecord,
    SessionCompaction,
    SessionContextWindow,
    TokenUsage,
)
from iris.message import Msg, ToolUseBlock
from iris.runtime import RuntimeCursor
from iris.runtime.commit import (
    RuntimeCompactionCommit,
    RuntimeModelStepCommit,
    RuntimeRunInputCommit,
    RuntimeToolCall,
    RuntimeToolResultCommit,
)
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolResult

NOW = datetime(2026, 7, 29, tzinfo=UTC)
FINGERPRINT = "a" * 64


def _store_commit_port(
    *,
    event_collector: _RunEventCollector | None = None,
) -> tuple[InMemoryLifecycleStore, StoreRuntimeCommitPort, RuntimeToolCall]:
    store = InMemoryLifecycleStore()
    initial = RuntimeCursor(position="before_input", step_index=0)
    before = initial.model_copy(update={"position": "before_model"})
    created = store.create_run(
        CreateRun(
            request=AgentRunRequest(input="hello", session_id="session_1", run_id="run_1"),
            options=AgentRunOptions(),
            agent_id="agent_1",
            start_activation_id="activation_1",
            initial_checkpoint=RunCheckpoint(
                run_id="run_1",
                sequence=1,
                activation_id="activation_1",
                engine_cursor=initial.model_dump(mode="json"),
                session_revision=0,
                model_steps_reserved=0,
                model_steps_committed=0,
            ),
            now=NOW,
        )
    )
    port = StoreRuntimeCommitPort(
        workspace_root=Path("workspace"),
        store=store,
        run=created.run,
        activation_id="activation_1",
        cursor=initial,
        clock=lambda: NOW,
        event_collector=event_collector or _RunEventCollector(),
    )
    port.commit_run_input(
        RuntimeRunInputCommit(
            cursor_before=initial,
            message_delta=(Msg.user("hello"),),
            cursor_after=before,
            initial_context_window=SessionContextWindow(),
        )
    )
    port.reserve_model_step(before)
    tool_use = ToolUseBlock(id="call_1", name="echo", input={"value": "hello"})
    assistant = Msg.assistant([tool_use])
    call = RuntimeToolCall(
        run_id="run_1",
        activation_id="activation_1",
        step_index=0,
        ordinal=1,
        tool_call_id=tool_use.id,
        tool_name=tool_use.name,
        arguments={"value": "hello"},
        fingerprint=FINGERPRINT,
    )
    port.commit_model_step(
        RuntimeModelStepCommit(
            cursor_before=before,
            message_delta=(assistant,),
            assistant_message=assistant,
            prepared_tool_calls=(call,),
            cursor_after=RuntimeCursor(
                position="tool_batch",
                step_index=0,
                tool_calls=(tool_use,),
                assistant_message=assistant,
            ),
        )
    )
    return store, port, call


def _control_snapshot(port: StoreRuntimeCommitPort) -> RunControlSnapshot:
    run = port.run
    return RunControlSnapshot(
        run_id=run.run_id,
        session_id=run.session_id,
        phase=run.phase,
        revision=run.revision,
        current_activation_id=run.current_activation_id,
        cancellation_requested_at=run.cancellation_requested_at,
        cancellation_reason=run.cancellation_reason,
        last_event_sequence=run.last_event_sequence,
        updated_at=run.updated_at,
    )


def _request_cancel(store: InMemoryLifecycleStore) -> RunCommit:
    current = store.load_run("run_1")
    assert current is not None
    return store.request_cancellation(
        RequestCancellation(
            run_id="run_1",
            expected_run_revision=current.revision,
            activation_id="activation_1",
            reason="user requested",
            now=NOW,
        )
    )


def test_store_commit_port_relays_only_new_committed_events() -> None:
    relayed: list[RunEvent] = []
    collector = _RunEventCollector(relayed.append)

    _store_commit_port(event_collector=collector)

    assert relayed == collector.events
    assert relayed
    assert len({(event.run_id, event.sequence) for event in relayed}) == len(relayed)


def test_store_commit_port_does_not_relay_failed_store_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relayed: list[RunEvent] = []
    collector = _RunEventCollector(relayed.append)
    store, port, call = _store_commit_port(
        event_collector=collector,
    )
    before = list(relayed)

    def fail_claim(command: object) -> object:
        del command
        raise IrisRunPersistenceError("模拟 store mutation 失败")

    monkeypatch.setattr(store, "claim_tool_call", fail_claim)

    with pytest.raises(IrisRunPersistenceError, match="store mutation"):
        port.claim_tool_call(call)

    assert relayed == before


def test_store_commit_port_exact_control_read_does_not_load_events_or_mutate_local_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, _ = _store_commit_port()
    local = port.run
    calls = {"control": 0, "events": 0, "full_run": 0}
    original_control = store.load_run_control
    original_events = store.list_events
    original_full_run = store.load_run

    def load_control(run_id: str) -> RunControlSnapshot | None:
        calls["control"] += 1
        return original_control(run_id)

    def list_events(
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]:
        calls["events"] += 1
        return original_events(run_id, after_sequence, limit=limit)

    def load_run(run_id: str) -> RunRecord | None:
        calls["full_run"] += 1
        return original_full_run(run_id)

    monkeypatch.setattr(store, "load_run_control", load_control)
    monkeypatch.setattr(store, "list_events", list_events)
    monkeypatch.setattr(store, "load_run", load_run)

    assert port.cancellation_requested() is False
    assert port.run is local
    assert calls == {"control": 1, "events": 0, "full_run": 0}


def test_store_commit_port_accepts_and_relays_exact_external_cancellation() -> None:
    relayed: list[RunEvent] = []
    collector = _RunEventCollector(relayed.append)
    store, port, _ = _store_commit_port(
        event_collector=collector,
    )
    before = len(collector.events)
    cancelled = _request_cancel(store)

    assert port.cancellation_requested() is True
    assert port.run == cancelled.run
    assert collector.events[before:] == list(cancelled.events)
    assert relayed == collector.events


def test_commit_port_cancellation_does_not_rescan_collected_event_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _RunEventCollector()
    store, port, _ = _store_commit_port(event_collector=collector)
    prior_ids = {id(event) for event in collector.events}
    cancelled = _request_cancel(store)
    prior_sequence_reads = 0
    original_getattribute = RunEvent.__getattribute__

    def getattribute(event: RunEvent, name: str) -> object:
        nonlocal prior_sequence_reads
        if name == "sequence" and id(event) in prior_ids:
            prior_sequence_reads += 1
        return original_getattribute(event, name)

    with monkeypatch.context() as patch:
        patch.setattr(RunEvent, "__getattribute__", getattribute)
        assert port.cancellation_requested()

    assert prior_sequence_reads == 0
    assert collector.events[-1] == cancelled.events[-1]


def test_store_commit_port_observes_cancellation_from_second_sqlite_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-process.db"
    owner = SQLiteStore(path)
    before = RuntimeCursor(position="before_input", step_index=0)
    created = owner.create_run(
        CreateRun(
            request=AgentRunRequest(input="hello", session_id="session_1", run_id="run_1"),
            options=AgentRunOptions(),
            agent_id="agent_1",
            start_activation_id="activation_1",
            initial_checkpoint=RunCheckpoint(
                run_id="run_1",
                sequence=1,
                activation_id="activation_1",
                engine_cursor=before.model_dump(mode="json"),
                session_revision=0,
                model_steps_reserved=0,
                model_steps_committed=0,
            ),
            now=NOW,
        )
    )
    collector = _RunEventCollector()
    collector.record(created.events)
    port = StoreRuntimeCommitPort(
        workspace_root=Path("workspace"),
        store=owner,
        run=created.run,
        activation_id="activation_1",
        cursor=before,
        clock=lambda: NOW,
        event_collector=collector,
    )
    remote = SQLiteStore(path)
    cancelled = remote.request_cancellation(
        RequestCancellation(
            run_id="run_1",
            expected_run_revision=created.run.revision,
            activation_id="activation_1",
            reason="remote",
            now=NOW,
        )
    )

    assert port.cancellation_requested() is True
    assert port.run == cancelled.run
    assert collector.events == owner.list_events("run_1")


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "other-run"},
        {"revision": 99},
        {"last_event_sequence": 99},
        {"phase": RunPhase.WAITING, "current_activation_id": None},
        {"current_activation_id": "other-activation"},
        {
            "revision": 4,
            "last_event_sequence": 5,
            "cancellation_requested_at": NOW,
            "cancellation_reason": "user requested",
            "updated_at": NOW - timedelta(seconds=1),
        },
    ],
    ids=[
        "run-identity",
        "revision-jump",
        "sequence-jump",
        "phase-change",
        "activation-change",
        "updated-at-backwards",
    ],
)
def test_store_commit_port_rejects_non_cancellation_control_changes(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
) -> None:
    store, port, _ = _store_commit_port()
    snapshot = RunControlSnapshot.model_validate(_control_snapshot(port).model_dump() | changes)
    monkeypatch.setattr(store, "load_run_control", lambda run_id: snapshot)

    with pytest.raises(IrisRunConflictError, match="cancellation mutation"):
        port.cancellation_requested()


def test_store_commit_port_uses_point_reads_for_single_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, call = _store_commit_port()
    calls = {"point": 0, "list": 0}
    original_point = store.load_tool_call
    original_list = store.list_tool_calls

    def load_tool_call(run_id: str, tool_call_id: str) -> RunToolCallRecord | None:
        calls["point"] += 1
        return original_point(run_id, tool_call_id)

    def list_tool_calls(run_id: str, *, step_index: int | None = None) -> list[RunToolCallRecord]:
        calls["list"] += 1
        return original_list(run_id, step_index=step_index)

    monkeypatch.setattr(store, "load_tool_call", load_tool_call)
    monkeypatch.setattr(store, "list_tool_calls", list_tool_calls)

    port.claim_tool_call(call)

    assert calls == {"point": 1, "list": 0}


def test_store_commit_port_lists_existing_calls_once_per_prepared_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, call = _store_commit_port()
    steps: list[int | None] = []
    original_list = store.list_tool_calls
    second = replace(call, ordinal=2, tool_call_id="call_2", arguments={"value": "second"})

    def list_tool_calls(run_id: str, *, step_index: int | None = None) -> list[RunToolCallRecord]:
        steps.append(step_index)
        return original_list(run_id, step_index=step_index)

    monkeypatch.setattr(store, "list_tool_calls", list_tool_calls)

    prepared = port._new_prepared_records((call, second), now=NOW)

    assert [record.tool_call_id for record in prepared] == ["call_2"]
    assert steps == [call.step_index]


def test_store_commit_port_maps_same_activation_cancel_claim_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relayed: list[RunEvent] = []
    collector = _RunEventCollector(relayed.append)
    store, port, call = _store_commit_port(
        event_collector=collector,
    )
    original_claim = store.claim_tool_call

    def cancel_then_claim(command: object) -> object:
        current = store.load_run("run_1")
        assert current is not None
        cancelled = store.request_cancellation(
            RequestCancellation(
                run_id="run_1",
                expected_run_revision=current.revision,
                activation_id="activation_1",
                reason="user requested",
                now=NOW,
            )
        )
        collector.record(cancelled.events)
        return original_claim(command)

    monkeypatch.setattr(store, "claim_tool_call", cancel_then_claim)

    with pytest.raises(IrisCancellationRequestedError) as error:
        port.claim_tool_call(call)

    assert isinstance(error.value.__cause__, IrisRunConflictError | IrisRunStateError)
    assert port.run.cancellation_requested_at == NOW
    assert store.list_tool_calls("run_1")[0].phase == "prepared"
    assert relayed == collector.events
    assert len({(event.run_id, event.sequence) for event in relayed}) == len(relayed)


def test_compaction_usage_refreshes_control_without_events_and_survives_model_commit() -> None:
    """无 event 的自身 revision 推进不能误判为外来 mutation。"""
    store, port, call = _store_commit_port()
    checkpoint = port.checkpoint
    session = port.load_session()
    before = port.run
    usage = TokenUsage(input_tokens=20_000, output_tokens=2_000, total_tokens=22_000)

    port.record_compaction_usage(usage)
    port.record_compaction_usage(usage)

    assert not port.cancellation_requested()
    assert port.run.revision == before.revision + 2
    assert port.run.last_event_sequence == before.last_event_sequence
    assert port.checkpoint == checkpoint
    assert port.load_session() == session
    assert port.run.usage.compaction.total_tokens == 44_000
    claim = port.claim_tool_call(call)
    result = ToolResult(tool_use_id=call.tool_call_id, tool_name=call.tool_name, content=[])
    cursor = RuntimeCursor(position="before_model", step_index=1)
    port.commit_tool_result(
        RuntimeToolResultCommit(
            tool_call=call,
            claim=claim,
            result=result,
            message_delta=(result.to_msg(),),
            cursor_after=cursor,
        )
    )
    assert port.reserve_model_step(cursor).granted
    assistant = Msg.assistant("done")
    port.commit_model_step(
        RuntimeModelStepCommit(
            cursor_before=cursor,
            assistant_message=assistant,
            message_delta=(assistant,),
            cursor_after=RuntimeCursor(
                position="outcome_ready", step_index=1, assistant_message=assistant
            ),
            input_tokens=25_000,
            output_tokens=1_000,
            total_tokens=26_000,
        )
    )
    assert port.run.usage.total_tokens == 26_000
    assert port.run.usage.compaction.total_tokens == 44_000
    assert store.load_run(port.run.run_id) == port.run


def _reserved_compaction_port() -> tuple[InMemoryLifecycleStore, StoreRuntimeCommitPort]:
    """完成一组工具后预留下一模型步。"""
    store, port, call = _store_commit_port()
    claim = port.claim_tool_call(call)
    result = ToolResult(tool_use_id=call.tool_call_id, tool_name=call.tool_name, content=[])
    cursor = RuntimeCursor(position="before_model", step_index=1)
    port.commit_tool_result(
        RuntimeToolResultCommit(
            tool_call=call,
            claim=claim,
            result=result,
            message_delta=(result.to_msg(),),
            cursor_after=cursor,
        )
    )
    assert port.reserve_model_step(cursor).granted
    return store, port


def test_compaction_commit_keeps_cursor_and_pending_reservation() -> None:
    """摘要提交推进 checkpoint，重建 port 后仅复用原 pending reservation。"""
    store, port = _reserved_compaction_port()
    before = port.load_session()
    cursor = port.cursor
    sequence = port.checkpoint.sequence
    usage = port.run.usage
    compaction = SessionCompaction(summary="已完成工具调用", covered_message_count=3)

    assert (
        port.commit_compaction(
            RuntimeCompactionCommit(
                cursor_before=cursor,
                expected_session_revision=before.revision,
                compaction=compaction,
                context_window=SessionContextWindow(),
                before_input_tokens=80_000,
                after_input_tokens=20_000,
            )
        )
        == cursor
    )

    after = port.load_session()
    assert after.messages == before.messages
    assert after.compaction == compaction
    assert after.revision == before.revision + 1
    assert port.checkpoint.sequence == sequence + 1
    assert port.checkpoint.session_revision == after.revision
    assert port.run.usage == usage
    # 同一进程提交摘要后不能把已消费的复用标记重新打开。
    assert port._reusable_model_reservation is False

    resumed = StoreRuntimeCommitPort(
        store=store,
        run=port.run,
        activation_id="activation_1",
        cursor=cursor,
        clock=lambda: NOW,
        event_collector=_RunEventCollector(),
        workspace_root=Path("workspace"),
    )
    revision = resumed.run.revision
    assert resumed.reserve_model_step(cursor).granted
    assert resumed.run.revision == revision
    assert resumed.run.usage == usage


def test_compaction_usage_accepts_later_cancellation_and_rejects_projection() -> None:
    """摘要费用记录后仍能观察取消，取消后不能安装候选摘要。"""
    store, port = _reserved_compaction_port()
    session = port.load_session()
    port.record_compaction_usage(TokenUsage(total_tokens=22_000))
    _request_cancel(store)
    assert port.cancellation_requested()
    with pytest.raises(IrisRunStateError):
        port.commit_compaction(
            RuntimeCompactionCommit(
                cursor_before=port.cursor,
                expected_session_revision=session.revision,
                compaction=SessionCompaction(summary="摘要", covered_message_count=3),
                context_window=SessionContextWindow(),
                before_input_tokens=80_000,
                after_input_tokens=20_000,
            )
        )
    assert port.load_session() == session
    assert port.run.usage.compaction.total_tokens == 22_000
