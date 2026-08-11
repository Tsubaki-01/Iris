from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iris.exceptions import IrisCancellationRequestedError, IrisRunConflictError, IrisRunStateError
from iris.harness._commit_port import StoreRuntimeCommitPort
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    RequestCancellation,
    RunCheckpoint,
)
from iris.message import Msg, ToolUseBlock
from iris.runtime import RuntimeCursor
from iris.runtime.commit import RuntimeModelStepCommit, RuntimeToolCall
from iris.store import InMemoryLifecycleStore

NOW = datetime(2026, 7, 29, tzinfo=UTC)
FINGERPRINT = "a" * 64


def _store_commit_port() -> tuple[InMemoryLifecycleStore, StoreRuntimeCommitPort, RuntimeToolCall]:
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
        event_sink=[],
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


def test_store_commit_port_maps_same_activation_cancel_claim_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port, call = _store_commit_port()
    original_claim = store.claim_tool_call

    def cancel_then_claim(command: object) -> object:
        current = store.load_run("run_1")
        assert current is not None
        store.request_cancellation(
            RequestCancellation(
                run_id="run_1",
                expected_run_revision=current.revision,
                activation_id="activation_1",
                reason="user requested",
                now=NOW,
            )
        )
        return original_claim(command)

    monkeypatch.setattr(store, "claim_tool_call", cancel_then_claim)

    with pytest.raises(IrisCancellationRequestedError) as error:
        port.claim_tool_call(call)

    assert isinstance(error.value.__cause__, IrisRunConflictError | IrisRunStateError)
    assert port.run.cancellation_requested_at == NOW
    assert store.list_tool_calls("run_1")[0].phase == "prepared"
