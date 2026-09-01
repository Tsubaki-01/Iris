"""Live streaming public boundary models 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from iris.streaming.models import (
    CommandReceipt,
    CommandRejected,
    LiveCursor,
    LiveEnvelope,
    LiveSubscriptionRequest,
    SubscribeAccepted,
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
