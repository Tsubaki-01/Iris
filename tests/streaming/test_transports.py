"""SSE 与 WebSocket framework-neutral adapter 测试。"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

import pytest

from iris.harness import SubmitReceipt
from iris.streaming.gateway import GatewaySubscription, StreamingGateway
from iris.streaming.models import (
    CommandReceipt,
    CommandRejected,
    DurableSync,
    DurableSyncItem,
    GatewayCommand,
    GatewayStreamItem,
    LiveEnvelope,
    ReplayGap,
    SubmitAccepted,
    SubmitCommand,
    SubscribeCommand,
    SubscriptionTerminal,
    SyncAccepted,
    SyncCommand,
)
from iris.streaming.sse import SSEAdapter
from iris.streaming.websocket import WebSocketAdapter


class FakeSubscription(AsyncIterator[GatewayStreamItem]):
    """测试 adapter cleanup 与有序输出的 subscription。"""

    def __init__(
        self,
        *items: GatewayStreamItem,
        stream_epoch: str = "epoch-1",
        scope: str = "session",
        scope_id: str = "session-1",
    ) -> None:
        self.items = deque(items)
        self.stream_epoch = stream_epoch
        self.scope = scope
        self.scope_id = scope_id
        self.closed = False

    def __aiter__(self) -> FakeSubscription:
        return self

    async def __anext__(self) -> GatewayStreamItem:
        if self.items:
            return self.items.popleft()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class DelayedSubscription(FakeSubscription):
    """在显式放行前保持同一个 pending anext。"""

    def __init__(self, item: GatewayStreamItem) -> None:
        super().__init__()
        self.item = item
        self.release = asyncio.Event()
        self.calls = 0
        self.cancelled = 0

    async def __anext__(self) -> GatewayStreamItem:
        self.calls += 1
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.calls == 1:
            return self.item
        raise StopAsyncIteration


class FakeGateway:
    """记录 WebSocket adapter 的 typed gateway 调用。"""

    def __init__(self, subscription: FakeSubscription) -> None:
        self.subscription = subscription
        self.session_id = "session-1"
        self.subscriptions: list[SubscribeCommand] = []
        self.commands: list[GatewayCommand] = []

    def subscribe(self, request: SubscribeCommand) -> GatewaySubscription:
        self.subscriptions.append(request)
        return cast(GatewaySubscription, self.subscription)

    async def handle(self, command: GatewayCommand) -> CommandReceipt:
        self.commands.append(command)
        if isinstance(command, SubmitCommand):
            return SubmitAccepted(
                request_id=command.request_id,
                receipt=SubmitReceipt(
                    submission_id=f"submission-{len(self.commands)}",
                    run_id="run-1",
                    mode=None,
                    state="delivered",
                ),
            )
        if isinstance(command, SyncCommand):
            return SyncAccepted(
                request_id=command.request_id,
                sync=DurableSync(session_id="session-1"),
            )
        return CommandRejected(
            request_id=command.request_id,
            command_kind=command.kind,
            code="TEST_REJECTED",
            message="测试拒绝",
        )


def _envelope(sequence: int = 1) -> LiveEnvelope:
    return LiveEnvelope(
        stream_epoch="epoch-1",
        scope="session",
        scope_id="session-1",
        live_sequence=sequence,
        kind="model.block.delta",
        run_id="run-1",
        session_id="session-1",
        activation_id="activation-1",
        payload={"channel": "text", "delta": "ok", "snapshot": "ok"},
    )


def _receiver(*frames: str | bytes | None) -> Callable[[], Awaitable[str | bytes | None]]:
    queued = deque(frames)

    async def receive() -> str | bytes | None:
        await asyncio.sleep(0.01)
        return queued.popleft() if queued else None

    return receive


@pytest.mark.asyncio
async def test_sse_encodes_envelope_and_control_items_without_fake_ids() -> None:
    subscription = FakeSubscription(
        _envelope(),
        ReplayGap(reason="unknown_cursor", current_epoch="epoch-1"),
        DurableSyncItem(sync=DurableSync(session_id="session-1")),
        SubscriptionTerminal(reason="broker_closed", message="closed"),
    )
    frames = [frame async for frame in SSEAdapter(heartbeat_interval_s=1).stream(subscription)]

    assert frames[0].startswith(b"id: {")
    assert b"event: model.block.delta\n" in frames[0]
    assert b"data: {" in frames[0]
    assert all(frame.endswith(b"\n\n") for frame in frames)
    assert b"id:" not in frames[1]
    assert b"event: replay.gap\n" in frames[1]
    assert b"id:" not in frames[2]
    assert b"event: sync.snapshot\n" in frames[2]
    assert b"id:" not in frames[3]
    assert subscription.closed


@pytest.mark.asyncio
async def test_sse_heartbeat_reuses_pending_anext_and_cleanup_only_closes() -> None:
    subscription = DelayedSubscription(_envelope())
    stream = SSEAdapter(heartbeat_interval_s=0.01).stream(subscription)

    heartbeat = await anext(stream)

    assert heartbeat == b": heartbeat\n\n"
    assert subscription.calls == 1
    assert subscription.cancelled == 0
    subscription.release.set()
    data = await anext(stream)
    assert b"event: model.block.delta" in data
    await stream.aclose()
    assert subscription.closed


@pytest.mark.asyncio
async def test_websocket_sync_first_then_subscribe_and_stream() -> None:
    subscription = FakeSubscription(_envelope())
    gateway = FakeGateway(subscription)
    sent: list[str] = []

    async def send(value: str) -> None:
        sent.append(value)

    frames = deque(
        (
            SyncCommand(request_id="sync-1").model_dump_json(),
            SubscribeCommand(
                request_id="subscribe-1",
                scope="session",
                scope_id="session-1",
            ).model_dump_json(),
        )
    )

    async def receive() -> str | None:
        if frames:
            await asyncio.sleep(0)
            return frames.popleft()
        while not any('"kind":"model.block.delta"' in value for value in sent):
            await asyncio.sleep(0)
        return None

    await WebSocketAdapter(gateway=cast(StreamingGateway, gateway)).serve(
        receive,
        send,
    )

    payloads = [json.loads(value) for value in sent]
    events = [payload["event"] for payload in payloads if "event" in payload]
    assert events.index("command.sync.accepted") < events.index("command.subscribe.accepted")
    assert any(payload.get("kind") == "model.block.delta" for payload in payloads)
    assert subscription.closed


@pytest.mark.asyncio
async def test_websocket_uses_one_writer_and_closes_on_receipt_backpressure() -> None:
    subscription = FakeSubscription()
    gateway = FakeGateway(subscription)
    release = asyncio.Event()
    send_started = asyncio.Event()
    concurrent = 0
    max_concurrent = 0

    async def slow_send(value: str) -> None:
        nonlocal concurrent, max_concurrent
        del value
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        send_started.set()
        await release.wait()
        concurrent -= 1

    task = asyncio.create_task(
        WebSocketAdapter(gateway=cast(StreamingGateway, gateway)).serve(
            _receiver(
                SubscribeCommand(
                    request_id="subscribe-1",
                    scope="session",
                    scope_id="session-1",
                ).model_dump_json(),
                SyncCommand(request_id="sync-1").model_dump_json(),
                SyncCommand(request_id="sync-2").model_dump_json(),
                SyncCommand(request_id="sync-3").model_dump_json(),
                None,
            ),
            slow_send,
        )
    )
    await asyncio.wait_for(send_started.wait(), timeout=1)
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(task, timeout=1)

    assert max_concurrent == 1
    assert subscription.closed


@pytest.mark.asyncio
async def test_websocket_disconnect_closes_observation_without_cancel_command() -> None:
    subscription = FakeSubscription()
    gateway = FakeGateway(subscription)

    await WebSocketAdapter(gateway=cast(StreamingGateway, gateway)).serve(
        _receiver(
            SubscribeCommand(
                request_id="subscribe-1",
                scope="session",
                scope_id="session-1",
            ).model_dump_json(),
            None,
        ),
        lambda value: asyncio.sleep(0),
    )

    assert subscription.closed
    assert all(command.kind != "cancel" for command in gateway.commands)


@pytest.mark.asyncio
async def test_websocket_send_failure_drains_children_and_closes_subscription() -> None:
    subscription = FakeSubscription(_envelope())
    gateway = FakeGateway(subscription)

    async def fail_send(value: str) -> None:
        del value
        raise RuntimeError("send failed")

    await WebSocketAdapter(gateway=cast(StreamingGateway, gateway)).serve(
        _receiver(
            SubscribeCommand(
                request_id="subscribe-1",
                scope="session",
                scope_id="session-1",
            ).model_dump_json(),
            None,
        ),
        fail_send,
    )

    assert subscription.closed
