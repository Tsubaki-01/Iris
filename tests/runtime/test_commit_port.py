"""Phase 2 runtime cursor 与 required commit port 契约测试。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fakes import FakeRuntimeCommitPort, resume_activation, start_activation
from pydantic import ValidationError

from iris.exceptions import IrisRunConflictError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    PermissionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import RunCheckpoint, RunErrorInfo, RuntimeExecutionOptions
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.runtime import (
    ModelStepReservation,
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeCursor,
)
from iris.runtime.commit import (
    RuntimeModelStepCommit,
    RuntimeSuspension,
    RuntimeToolCall,
    RuntimeToolResultCommit,
)
from iris.tools import ToolResult

NOW = datetime(2026, 7, 29, tzinfo=UTC)
FINGERPRINT = "a" * 64


def _tool_call(call_id: str = "call_1") -> ToolUseBlock:
    return ToolUseBlock(id=call_id, name="echo", input={"value": "hello"})


def _interaction() -> HumanInteraction:
    return HumanInteraction(
        interaction_id="interaction_1",
        session_id="session_1",
        run_id="run_1",
        step_index=0,
        status=InteractionStatus.PENDING,
        checkpoint={},
        request=HumanInteractionRequest(
            tool_call=ToolCallSnapshot(
                tool_call_id="call_1",
                tool_name="echo",
                arguments={"value": "hello"},
                workspace_root="J:/workspace",
                fingerprint=FINGERPRINT,
            ),
            prompt=PermissionPrompt(reason="需要确认"),
        ),
        created_at=NOW,
    )


def test_runtime_cursor_round_trips_through_lifecycle_checkpoint() -> None:
    assistant = Msg.assistant([_tool_call()])
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        next_tool_index=0,
        tool_calls=(_tool_call(),),
        assistant_message=assistant,
        read_state={"files": {}},
    )

    checkpoint = RunCheckpoint(
        run_id="run_1",
        sequence=1,
        activation_id="activation_1",
        engine_cursor=cursor.model_dump(mode="json"),
        session_revision=0,
        model_steps_reserved=1,
        model_steps_committed=1,
        environment_fingerprint=FINGERPRINT,
    )

    assert RuntimeCursor.model_validate(checkpoint.engine_cursor) == cursor


@pytest.mark.parametrize(
    "payload",
    [
        {
            "position": "before_model",
            "step_index": 0,
            "next_tool_index": 1,
            "tool_calls": [_tool_call().model_dump(mode="json")],
        },
        {
            "position": "tool_batch",
            "step_index": 0,
            "next_tool_index": 0,
            "tool_calls": [_tool_call().model_dump(mode="json")],
            "assistant_message": None,
        },
        {
            "position": "outcome_ready",
            "step_index": 0,
            "next_tool_index": 0,
            "tool_calls": [_tool_call().model_dump(mode="json")],
            "assistant_message": Msg.assistant([TextBlock(text="done")]).model_dump(mode="json"),
        },
    ],
)
def test_runtime_cursor_rejects_impossible_positions(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RuntimeCursor.model_validate(payload)


def test_runtime_activation_input_enforces_start_and_resume_shape() -> None:
    cursor = RuntimeCursor(position="before_model", step_index=0)

    start = RuntimeActivationInput(
        run_id="run_1",
        activation_id="activation_1",
        session_id="session_1",
        kind="start",
        input="hello",
        cursor=cursor,
        options=RuntimeExecutionOptions(),
    )

    assert start.input == "hello"
    with pytest.raises(ValidationError):
        RuntimeActivationInput(
            run_id="run_1",
            activation_id="activation_1",
            session_id="session_1",
            kind="resume",
            input="unexpected",
            cursor=cursor,
            options=RuntimeExecutionOptions(),
        )


def test_runtime_activation_interaction_projection_is_exact_and_json_safe() -> None:
    assistant = Msg.assistant([_tool_call()])
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        tool_calls=tuple(assistant.tool_calls),
        assistant_message=assistant,
    )
    approval = RuntimeApprovedToolCall(
        interaction_id="interaction_1",
        tool_call_id="call_1",
        tool_name="echo",
        fingerprint=FINGERPRINT,
    )
    activation = resume_activation(cursor, interaction_projection=approval)

    assert activation.interaction_projection == approval
    assert RuntimeActivationInput.model_validate(
        activation.model_dump(mode="json")
    ) == activation
    with pytest.raises(ValidationError, match="start activation"):
        RuntimeActivationInput(
            run_id="run_1",
            activation_id="activation_1",
            session_id="session_1",
            kind="start",
            input="hello",
            cursor=RuntimeCursor(position="before_model", step_index=0),
            options=RuntimeExecutionOptions(),
            interaction_projection=approval,
        )


def test_runtime_activation_result_enforces_outcome_payloads() -> None:
    cursor = RuntimeCursor(position="before_model", step_index=1)
    error = RunErrorInfo(code="PROVIDER_ERROR", message="failed", source="provider")

    failed = RuntimeActivationResult(
        outcome=RuntimeActivationOutcome.FAILED,
        cursor=cursor,
        error=error,
    )
    suspended = RuntimeActivationResult(
        outcome=RuntimeActivationOutcome.SUSPENDED,
        cursor=cursor,
        suspension=_interaction(),
    )

    assert failed.error == error
    assert suspended.suspension is not None
    with pytest.raises(ValidationError):
        RuntimeActivationResult(
            outcome=RuntimeActivationOutcome.COMPLETED,
            cursor=cursor,
            error=error,
        )


def test_commit_models_are_frozen_and_preserve_exact_cursor_facts() -> None:
    before = RuntimeCursor(position="before_model", step_index=0)
    after = RuntimeCursor(
        position="outcome_ready",
        step_index=0,
        assistant_message=Msg.assistant("done"),
    )
    prepared = RuntimeToolCall(
        run_id="run_1",
        activation_id="activation_1",
        step_index=0,
        ordinal=1,
        tool_call_id="call_1",
        tool_name="echo",
        arguments={"value": "hello"},
        fingerprint=FINGERPRINT,
    )
    commit = RuntimeModelStepCommit(
        cursor_before=before,
        message_delta=(Msg.user("hello"), Msg.assistant("done")),
        assistant_message=Msg.assistant("done"),
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        prepared_tool_calls=(prepared,),
        cursor_after=after,
    )
    suspension = RuntimeSuspension(
        cursor_before=before,
        message_delta=(Msg.user("hello"), Msg.assistant([_tool_call()])),
        assistant_message=Msg.assistant([_tool_call()]),
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        prepared_tool_calls=(prepared,),
        cursor=RuntimeCursor(
            position="tool_batch",
            step_index=0,
            tool_calls=(_tool_call(),),
            assistant_message=Msg.assistant([_tool_call()]),
        ),
        interaction_request=_interaction().request,
        expires_at=NOW,
    )
    reservation = ModelStepReservation(
        granted=True,
        step_index=0,
        cursor=before,
        remaining_deadline_seconds=2.5,
    )

    assert commit.cursor_after == after
    assert suspension.interaction_request.prompt.kind.value == "permission"
    assert reservation.remaining_deadline_seconds == 2.5
    with pytest.raises(ValidationError):
        commit.input_tokens = 4


def test_runtime_cursor_rejects_live_message_metadata() -> None:
    with pytest.raises(ValidationError, match="JSON-safe"):
        RuntimeCursor(
            position="outcome_ready",
            step_index=0,
            assistant_message=Msg.assistant("done", metadata={"live": object()}),
        )


def test_commit_dto_rejects_live_tool_result_metadata() -> None:
    call = RuntimeToolCall(
        run_id="run_1",
        activation_id="activation_1",
        step_index=0,
        ordinal=1,
        tool_call_id="call_1",
        tool_name="echo",
        arguments={},
        fingerprint=FINGERPRINT,
    )
    with pytest.raises(ValidationError, match="JSON-safe"):
        RuntimeToolResultCommit(
            tool_call=call,
            result=ToolResult(
                tool_use_id="call_1",
                tool_name="echo",
                metadata={"live": object()},
            ),
            message_delta=(Msg.tool_result(tool_use_id="call_1"),),
            cursor_after=RuntimeCursor(position="before_model", step_index=1),
        )


def test_fake_commit_port_rejects_model_commit_without_reservation() -> None:
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation)
    assistant = Msg.assistant("done")

    with pytest.raises(IrisRunConflictError, match="reservation"):
        port.commit_model_step(
            RuntimeModelStepCommit(
                cursor_before=activation.cursor,
                assistant_message=assistant,
                cursor_after=RuntimeCursor(
                    position="outcome_ready",
                    step_index=0,
                    assistant_message=assistant,
                ),
            )
        )


def test_fake_commit_port_rejects_claim_for_non_current_call() -> None:
    assistant = Msg.assistant([_tool_call("call_1")])
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        tool_calls=tuple(assistant.tool_calls),
        assistant_message=assistant,
    )
    activation = resume_activation(cursor)
    port = FakeRuntimeCommitPort(activation, messages=[assistant])
    wrong = RuntimeToolCall(
        run_id=activation.run_id,
        activation_id=activation.activation_id,
        step_index=0,
        ordinal=1,
        tool_call_id="other_call",
        tool_name="echo",
        arguments={"value": "hello"},
        fingerprint=FINGERPRINT,
    )

    with pytest.raises(IrisRunConflictError, match="cursor"):
        port.claim_tool_call(wrong)


def test_fake_commit_port_rejects_result_that_skips_cursor_position() -> None:
    assistant = Msg.assistant([_tool_call("call_1")])
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        tool_calls=tuple(assistant.tool_calls),
        assistant_message=assistant,
    )
    activation = resume_activation(cursor)
    port = FakeRuntimeCommitPort(activation, messages=[assistant])
    call = RuntimeToolCall(
        run_id=activation.run_id,
        activation_id=activation.activation_id,
        step_index=0,
        ordinal=1,
        tool_call_id="call_1",
        tool_name="echo",
        arguments={"value": "hello"},
        fingerprint=FINGERPRINT,
    )
    result = ToolResult(tool_use_id="call_1", tool_name="echo")
    claim = port.claim_tool_call(call)

    with pytest.raises(IrisRunConflictError, match="推进一个 cursor"):
        port.commit_tool_result(
            RuntimeToolResultCommit(
                tool_call=call,
                claim=claim,
                result=result,
                message_delta=(Msg.tool_result(tool_use_id="call_1"),),
                cursor_after=RuntimeCursor(position="before_model", step_index=3),
            )
        )


def test_fake_commit_port_rejects_ordinary_success_without_claim() -> None:
    assistant = Msg.assistant([_tool_call("call_1")])
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=0,
        tool_calls=tuple(assistant.tool_calls),
        assistant_message=assistant,
    )
    activation = resume_activation(cursor)
    port = FakeRuntimeCommitPort(activation, messages=[assistant])
    call = RuntimeToolCall(
        run_id=activation.run_id,
        activation_id=activation.activation_id,
        step_index=0,
        ordinal=1,
        tool_call_id="call_1",
        tool_name="echo",
        arguments={"value": "hello"},
        fingerprint=FINGERPRINT,
    )

    with pytest.raises(IrisRunConflictError, match="缺少 durable claim"):
        port.commit_tool_result(
            RuntimeToolResultCommit(
                tool_call=call,
                result=ToolResult(tool_use_id="call_1", tool_name="echo"),
                message_delta=(Msg.tool_result(tool_use_id="call_1"),),
                cursor_after=RuntimeCursor(position="before_model", step_index=1),
            )
        )


def test_fake_commit_port_rejects_invalid_model_cursor_transition() -> None:
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation)
    port.reserve_model_step(activation.cursor)
    assistant = Msg.assistant("done")

    with pytest.raises(IrisRunConflictError, match="转换无效"):
        port.commit_model_step(
            RuntimeModelStepCommit(
                cursor_before=activation.cursor,
                message_delta=(Msg.user("hello"), assistant),
                assistant_message=assistant,
                cursor_after=RuntimeCursor(
                    position="outcome_ready",
                    step_index=1,
                    assistant_message=assistant,
                ),
            )
        )
