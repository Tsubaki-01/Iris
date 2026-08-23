from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.harness._commit_port import StoreRuntimeCommitPort
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    RequestCancellation,
    RunCheckpoint,
    RunCommit,
    RunControlSnapshot,
    RunEvent,
    RunEventKind,
    RunPhase,
)
from iris.message import Msg, ToolUseBlock
from iris.runtime import RuntimeCursor
from iris.runtime.commit import RuntimeModelStepCommit, RuntimeToolCall
from iris.store import InMemoryLifecycleStore, SQLiteStore

NOW = datetime(2026, 7, 29, tzinfo=UTC)
FINGERPRINT = "a" * 64


def _store_commit_port(
    *,
    event_sink: list[RunEvent] | None = None,
    durable_event_callback: Callable[[RunEvent], None] | None = None,
) -> tuple[InMemoryLifecycleStore, StoreRuntimeCommitPort, RuntimeToolCall]:
    store = InMemoryLifecycleStore()
    before = RuntimeCursor(position="before_model", step_index=0)
    created = store.create_run(
        CreateRun(
            request=AgentRunRequest(input="hello", session_id="session_1", run_id="run_1"),
            options=AgentRunOptions(),
            agent_id="agent_1",
            environment_fingerprint=FINGERPRINT,
            start_activation_id="activation_1",
            initial_checkpoint=RunCheckpoint(
                run_id="run_1",
                sequence=1,
                activation_id="activation_1",
                engine_cursor=before.model_dump(mode="json"),
                session_revision=0,
                model_steps_reserved=0,
                model_steps_committed=0,
                environment_fingerprint=FINGERPRINT,
            ),
            now=NOW,
        )
    )
    port = StoreRuntimeCommitPort(
        store=store,
        run=created.run,
        activation_id="activation_1",
        clock=lambda: NOW,
        event_sink=[] if event_sink is None else event_sink,
        durable_event_callback=durable_event_callback,
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
            message_delta=(Msg.user("hello"), assistant),
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
    collected: list[RunEvent] = []
    relayed: list[RunEvent] = []

    _store_commit_port(event_sink=collected, durable_event_callback=relayed.append)

    assert relayed == collected
    assert relayed
    assert len({(event.run_id, event.sequence) for event in relayed}) == len(relayed)


def test_store_commit_port_isolates_durable_event_callback_failure() -> None:
    collected: list[RunEvent] = []
    attempted: list[RunEvent] = []

    def raising_callback(event: RunEvent) -> None:
        attempted.append(event)
        raise RuntimeError("模拟 durable event callback 失败")

    store, port, _ = _store_commit_port(
        event_sink=collected,
        durable_event_callback=raising_callback,
    )

    assert attempted == collected
    assert port.run == store.load_run("run_1")


def test_store_commit_port_does_not_relay_failed_store_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collected: list[RunEvent] = []
    relayed: list[RunEvent] = []
    store, port, call = _store_commit_port(
        event_sink=collected,
        durable_event_callback=relayed.append,
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

    def load_control(run_id: str):
        calls["control"] += 1
        return original_control(run_id)

    def list_events(run_id: str, after_sequence: int = 0):
        calls["events"] += 1
        return original_events(run_id, after_sequence)

    def load_run(run_id: str):
        calls["full_run"] += 1
        return original_full_run(run_id)

    monkeypatch.setattr(store, "load_run_control", load_control)
    monkeypatch.setattr(store, "list_events", list_events)
    monkeypatch.setattr(store, "load_run", load_run)

    assert port.cancellation_requested() is False
    assert port.run is local
    assert calls == {"control": 1, "events": 0, "full_run": 0}


def test_store_commit_port_maps_missing_control_to_run_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, _ = _store_commit_port()
    monkeypatch.setattr(store, "load_run_control", lambda run_id: None)

    with pytest.raises(IrisRunNotFoundError, match="绑定的 run 不存在"):
        port.cancellation_requested()


def test_store_commit_port_accepts_and_relays_exact_external_cancellation() -> None:
    collected: list[RunEvent] = []
    relayed: list[RunEvent] = []
    store, port, _ = _store_commit_port(
        event_sink=collected,
        durable_event_callback=relayed.append,
    )
    before = len(collected)
    cancelled = _request_cancel(store)

    assert port.cancellation_requested() is True
    assert port.run == cancelled.run
    assert collected[before:] == list(cancelled.events)
    assert relayed == collected


def test_store_commit_port_observes_cancellation_from_second_sqlite_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-process.db"
    owner = SQLiteStore(path)
    before = RuntimeCursor(position="before_model", step_index=0)
    created = owner.create_run(
        CreateRun(
            request=AgentRunRequest(input="hello", session_id="session_1", run_id="run_1"),
            options=AgentRunOptions(),
            agent_id="agent_1",
            environment_fingerprint=FINGERPRINT,
            start_activation_id="activation_1",
            initial_checkpoint=RunCheckpoint(
                run_id="run_1",
                sequence=1,
                activation_id="activation_1",
                engine_cursor=before.model_dump(mode="json"),
                session_revision=0,
                model_steps_reserved=0,
                model_steps_committed=0,
                environment_fingerprint=FINGERPRINT,
            ),
            now=NOW,
        )
    )
    events = list(created.events)
    port = StoreRuntimeCommitPort(
        store=owner,
        run=created.run,
        activation_id="activation_1",
        clock=lambda: NOW,
        event_sink=events,
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
    assert events == owner.list_events("run_1")


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


@pytest.mark.parametrize(
    "variant",
    ["missing", "multiple", "kind", "sequence", "activation", "payload"],
)
def test_store_commit_port_rejects_unproven_cancellation_event(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    store, port, _ = _store_commit_port()
    cancelled = _request_cancel(store)
    event = cancelled.events[0]
    events = [event]
    if variant == "missing":
        events = []
    elif variant == "multiple":
        events = [event, event]
    elif variant == "kind":
        events = [event.model_copy(update={"kind": RunEventKind.MODEL_STEP_RESERVED})]
    elif variant == "sequence":
        events = [event.model_copy(update={"sequence": event.sequence + 1})]
    elif variant == "activation":
        events = [event.model_copy(update={"activation_id": "other-activation"})]
    else:
        events = [event.model_copy(update={"payload": {"reason": "other"}})]
    monkeypatch.setattr(store, "list_events", lambda run_id, after_sequence=0: events)

    with pytest.raises(IrisRunConflictError, match="cancellation event"):
        port.cancellation_requested()


def test_store_commit_port_rejects_second_control_change_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, _ = _store_commit_port()
    _request_cancel(store)
    assert port.cancellation_requested() is True
    current = _control_snapshot(port)
    repeated = RunControlSnapshot.model_validate(
        current.model_dump()
        | {
            "revision": current.revision + 1,
            "last_event_sequence": current.last_event_sequence + 1,
            "updated_at": NOW + timedelta(seconds=1),
        }
    )
    monkeypatch.setattr(store, "load_run_control", lambda run_id: repeated)

    with pytest.raises(IrisRunConflictError, match="cancellation mutation"):
        port.cancellation_requested()


def test_store_commit_port_uses_point_reads_for_single_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, call = _store_commit_port()
    calls = {"point": 0, "list": 0}
    original_point = store.load_tool_call
    original_list = store.list_tool_calls

    def load_tool_call(run_id: str, tool_call_id: str):
        calls["point"] += 1
        return original_point(run_id, tool_call_id)

    def list_tool_calls(run_id: str):
        calls["list"] += 1
        return original_list(run_id)

    monkeypatch.setattr(store, "load_tool_call", load_tool_call)
    monkeypatch.setattr(store, "list_tool_calls", list_tool_calls)

    port.claim_tool_call(call)

    assert calls == {"point": 1, "list": 0}


def test_store_commit_port_lists_existing_calls_once_per_prepared_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, call = _store_commit_port()
    calls = 0
    original_list = store.list_tool_calls
    second = call.model_copy(
        update={
            "ordinal": 2,
            "tool_call_id": "call_2",
            "arguments": {"value": "second"},
        }
    )

    def list_tool_calls(run_id: str):
        nonlocal calls
        calls += 1
        return original_list(run_id)

    monkeypatch.setattr(store, "list_tool_calls", list_tool_calls)

    prepared = port._new_prepared_records((call, second), now=NOW)

    assert [record.tool_call_id for record in prepared] == ["call_2"]
    assert calls == 1


def test_store_commit_port_maps_same_activation_cancel_claim_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collected: list[RunEvent] = []
    relayed: list[RunEvent] = []
    store, port, call = _store_commit_port(
        event_sink=collected,
        durable_event_callback=relayed.append,
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
        for event in cancelled.events:
            key = (event.run_id, event.sequence)
            if key not in {(item.run_id, item.sequence) for item in collected}:
                collected.append(event)
                relayed.append(event)
        return original_claim(command)

    monkeypatch.setattr(store, "claim_tool_call", cancel_then_claim)

    with pytest.raises(IrisCancellationRequestedError) as error:
        port.claim_tool_call(call)

    assert isinstance(error.value.__cause__, IrisRunConflictError | IrisRunStateError)
    assert port.run.cancellation_requested_at == NOW
    assert store.list_tool_calls("run_1")[0].phase == "prepared"
    assert relayed == collected
    assert len({(event.run_id, event.sequence) for event in relayed}) == len(relayed)
