"""Live streaming public boundary models 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from iris.streaming.models import (
    CancelCommand,
    CommandReceipt,
    CommandRejected,
    DurableRunCursor,
    GatewayCommand,
    LiveCursor,
    LiveEnvelope,
    LiveSubscriptionRequest,
    ReplayGap,
    SubmitAccepted,
    SubscribeAccepted,
    SubscribeCommand,
    SyncCommand,
)


def test_live_cursor_rejects_extra_blank_and_invalid_sequence() -> None:
    with pytest.raises(ValidationError, match="extra"):
        LiveCursor(
            stream_epoch="epoch-1",
            scope="run",
            scope_id="run-1",
            after_live_sequence=0,
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        LiveCursor(
            stream_epoch=" ",
            scope="run",
            scope_id="run-1",
            after_live_sequence=0,
        )
    with pytest.raises(ValidationError):
        LiveCursor(
            stream_epoch="epoch-1",
            scope="run",
            scope_id="run-1",
            after_live_sequence=-1,
        )
    with pytest.raises(ValidationError):
        LiveCursor(
            stream_epoch="epoch-1",
            scope="run",
            scope_id="run-1",
            after_live_sequence=True,
        )


def test_live_subscription_request_owns_cursor_scope_match() -> None:
    cursor = LiveCursor(
        stream_epoch="epoch-1",
        scope="run",
        scope_id="run-1",
        after_live_sequence=2,
    )

    request = LiveSubscriptionRequest(scope="run", scope_id="run-1", cursor=cursor)

    assert request.cursor == cursor
    with pytest.raises(ValidationError, match="cursor"):
        LiveSubscriptionRequest(scope="session", scope_id="session-1", cursor=cursor)


def test_live_envelope_requires_json_safe_payload() -> None:
    envelope = LiveEnvelope(
        stream_epoch="epoch-1",
        scope="run",
        scope_id="run-1",
        live_sequence=1,
        kind="model.block.delta",
        run_id="run-1",
        session_id="session-1",
        activation_id="activation-1",
        payload={"delta": "ok"},
    )

    assert envelope.live_sequence == 1
    with pytest.raises(ValidationError):
        LiveEnvelope(
            stream_epoch="epoch-1",
            scope="run",
            scope_id="run-1",
            live_sequence=1,
            kind="model.block.delta",
            payload={"path": Path("secret.txt")},
        )
    with pytest.raises(ValidationError):
        LiveEnvelope(
            stream_epoch="epoch-1",
            scope="run",
            scope_id="run-1",
            live_sequence=1,
            kind="model.block.delta",
            payload={"value": float("nan")},
        )


def test_replay_gap_has_no_live_sequence() -> None:
    gap = ReplayGap(
        reason="epoch_changed",
        requested_cursor=LiveCursor(
            stream_epoch="old",
            scope="run",
            scope_id="run-1",
            after_live_sequence=1,
        ),
        current_epoch="new",
    )

    assert gap.kind == "replay.gap"
    assert "live_sequence" not in gap.model_dump()


@pytest.mark.parametrize("command_type", [SubscribeCommand, SyncCommand])
def test_commands_reject_duplicate_durable_run_cursors(command_type: type) -> None:
    cursors = (
        DurableRunCursor(run_id="run-1", after_sequence=0),
        DurableRunCursor(run_id="run-1", after_sequence=2),
    )

    with pytest.raises(ValidationError, match="duplicate"):
        if command_type is SubscribeCommand:
            command_type(
                kind="subscribe",
                request_id="request-1",
                scope="session",
                scope_id="session-1",
                durable_cursors=cursors,
            )
        else:
            command_type(
                kind="sync",
                request_id="request-1",
                cursors=cursors,
            )


def test_gateway_command_discriminator_excludes_subscribe() -> None:
    adapter = TypeAdapter(GatewayCommand)

    cancel = adapter.validate_python(
        {"kind": "cancel", "request_id": "request-1", "reason": "用户取消"}
    )

    assert isinstance(cancel, CancelCommand)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "subscribe",
                "request_id": "request-2",
                "scope": "session",
                "scope_id": "session-1",
            }
        )


def test_command_receipt_discriminator_and_safe_rejection() -> None:
    adapter = TypeAdapter(CommandReceipt)

    accepted = adapter.validate_python(
        {
            "event": "command.subscribe.accepted",
            "request_id": "request-1",
            "stream_epoch": "epoch-1",
            "scope": "session",
            "scope_id": "session-1",
        }
    )
    rejected = adapter.validate_python(
        {
            "event": "command.rejected",
            "request_id": None,
            "command_kind": None,
            "code": "INVALID_COMMAND",
            "message": "命令无效",
        }
    )

    assert isinstance(accepted, SubscribeAccepted)
    assert isinstance(rejected, CommandRejected)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "event": "command.rejected",
                "request_id": "request-2",
                "command_kind": "submit",
                "code": "INVALID_COMMAND",
                "message": "命令无效",
                "traceback": "secret",
            }
        )


def test_submit_receipt_model_keeps_existing_typed_receipt() -> None:
    receipt = SubmitAccepted.model_validate(
        {
            "event": "command.submit.accepted",
            "request_id": "request-1",
            "receipt": {
                "submission_id": "submission-1",
                "run_id": "run-1",
                "mode": None,
                "state": "delivered",
            },
        }
    )

    assert receipt.receipt.run_id == "run-1"
