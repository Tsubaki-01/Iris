"""进程内 live broker 顺序、replay 与隔离测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.harness.session_manager import SubmissionEvent
from iris.harness.streaming import SessionSubmissionEvent
from iris.lifecycle import RunEvent, RunEventKind
from iris.message import (
    LLMResponse,
    ModelBlockDelta,
    ModelBlockRef,
    ModelResponseCompleted,
    ModelStreamScope,
    TextBlock,
)
from iris.runtime import RuntimeStreamEvent
from iris.streaming.broker import LiveStreamBroker, LiveSubscription
from iris.streaming.models import (
    LiveCursor,
    LiveEnvelope,
    LiveSubscriptionRequest,
    ReplayGap,
    SubscriptionTerminal,
)
from iris.streaming.projection import project_live_fact
from iris.tools import ToolArtifact, ToolErrorInfo, ToolResult


def _run_event(
    sequence: int,
    *,
    run_id: str = "run-1",
    session_id: str = "session-1",
    kind: RunEventKind = RunEventKind.RUN_STARTED,
) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        session_id=session_id,
        sequence=sequence,
        kind=kind,
        occurred_at=datetime.now(UTC),
        payload={"stop_reason": "completed"} if kind is RunEventKind.RUN_TERMINAL else {},
    )


def _submission_event(state: str = "pending") -> SessionSubmissionEvent:
    reason = "session_closed" if state == "failed" else None
    return SessionSubmissionEvent(
        session_id="session-1",
        event=SubmissionEvent(
            submission_id="submission-1",
            run_id="run-1",
            mode="steer",
            state=state,
            reason=reason,
        ),
    )


def _delta_fact(
    *,
    snapshot: str,
    block_id: str = "block-1",
    provider_sequence: int = 1,
) -> RuntimeStreamEvent:
    scope = ModelStreamScope(
        model_stream_id="model-stream-1",
        provider="fake",
        model="model-1",
        attempt=1,
    )
    block = ModelBlockRef(index=0, block_id=block_id, kind="text")
    model_event = ModelBlockDelta(
        scope=scope,
        sequence=provider_sequence,
        occurred_at=datetime.now(UTC),
        block=block,
        channel="text",
        delta=snapshot[-1],
        snapshot=snapshot,
    )
    return RuntimeStreamEvent(
        kind="model.event",
        run_id="run-1",
        session_id="session-1",
        activation_id="activation-1",
        step_index=0,
        model_event=model_event,
    )


async def _next(subscription: LiveSubscription):
    return await asyncio.wait_for(anext(subscription), timeout=0.5)


@pytest.mark.asyncio
async def test_run_and_session_sequences_are_independent() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=8, subscription_capacity=8)
    run_sub = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-1"))
    session_sub = broker.subscribe(LiveSubscriptionRequest(scope="session", scope_id="session-1"))

    broker.publish(_submission_event())
    session_submission = await _next(session_sub)
    broker.publish(_run_event(7))
    run_item = await _next(run_sub)
    session_run_item = await _next(session_sub)

    assert isinstance(session_submission, LiveEnvelope)
    assert session_submission.live_sequence == 1
    assert session_submission.durable_sequence is None
    assert isinstance(run_item, LiveEnvelope)
    assert run_item.live_sequence == 1
    assert run_item.durable_sequence == 7
    assert isinstance(session_run_item, LiveEnvelope)
    assert session_run_item.live_sequence == 2
    assert session_run_item.durable_sequence == 7


@pytest.mark.asyncio
async def test_valid_replay_handoff_has_no_duplicate_or_gap() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=4, subscription_capacity=4)
    broker.publish(_run_event(1))
    broker.publish(_run_event(2))
    cursor = LiveCursor(
        stream_epoch=broker.current_epoch(),
        scope="run",
        scope_id="run-1",
        after_live_sequence=1,
    )

    subscription = broker.subscribe(
        LiveSubscriptionRequest(scope="run", scope_id="run-1", cursor=cursor)
    )
    replayed = await _next(subscription)
    broker.publish(_run_event(3))
    live = await _next(subscription)

    assert isinstance(replayed, LiveEnvelope)
    assert isinstance(live, LiveEnvelope)
    assert [replayed.live_sequence, live.live_sequence] == [2, 3]


@pytest.mark.asyncio
async def test_epoch_unknown_ahead_and_expired_cursors_return_typed_gap() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=2, subscription_capacity=4)
    broker.publish(_run_event(1))
    broker.publish(_run_event(2))
    broker.publish(_run_event(3))
    epoch = broker.current_epoch()
    cursors = (
        (
            LiveCursor(
                stream_epoch="old-epoch",
                scope="run",
                scope_id="run-1",
                after_live_sequence=1,
            ),
            "epoch_changed",
        ),
        (
            LiveCursor(
                stream_epoch=epoch,
                scope="run",
                scope_id="run-1",
                after_live_sequence=99,
            ),
            "unknown_cursor",
        ),
        (
            LiveCursor(
                stream_epoch=epoch,
                scope="run",
                scope_id="run-1",
                after_live_sequence=0,
            ),
            "cursor_expired",
        ),
        (
            LiveCursor(
                stream_epoch=epoch,
                scope="run",
                scope_id="missing-run",
                after_live_sequence=0,
            ),
            "unknown_cursor",
        ),
    )

    for cursor, reason in cursors:
        subscription = broker.subscribe(
            LiveSubscriptionRequest(
                scope=cursor.scope,
                scope_id=cursor.scope_id,
                cursor=cursor,
            )
        )
        item = await _next(subscription)
        assert isinstance(item, ReplayGap)
        assert item.reason == reason
        assert "live_sequence" not in item.model_dump()


@pytest.mark.asyncio
async def test_same_partial_key_replaces_pending_item() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=8, subscription_capacity=2)
    subscription = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-1"))

    broker.publish(_delta_fact(snapshot="a", provider_sequence=1))
    broker.publish(_delta_fact(snapshot="ab", provider_sequence=2))
    item = await _next(subscription)

    assert isinstance(item, LiveEnvelope)
    assert item.live_sequence == 2
    assert item.payload["snapshot"] == "ab"


@pytest.mark.asyncio
async def test_critical_fact_evicts_pending_partial() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=8, subscription_capacity=1)
    subscription = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-1"))
    broker.publish(_delta_fact(snapshot="a"))

    broker.publish(_run_event(1))
    item = await _next(subscription)

    assert isinstance(item, LiveEnvelope)
    assert item.kind == RunEventKind.RUN_STARTED.value


@pytest.mark.asyncio
async def test_full_critical_queue_gaps_only_slow_subscriber() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=8, subscription_capacity=2)
    slow = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-1"))
    fast = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-1"))

    fast_sequences: list[int] = []
    for sequence in range(1, 4):
        broker.publish(_run_event(sequence))
        fast_item = await _next(fast)
        assert isinstance(fast_item, LiveEnvelope)
        fast_sequences.append(fast_item.live_sequence)
    broker.publish(_run_event(4))
    fourth = await _next(fast)

    gap = await _next(slow)
    terminal = await _next(slow)
    with pytest.raises(StopAsyncIteration):
        await slow.__anext__()

    assert fast_sequences == [1, 2, 3]
    assert isinstance(fourth, LiveEnvelope) and fourth.live_sequence == 4
    assert isinstance(gap, ReplayGap) and gap.reason == "slow_consumer"
    assert isinstance(terminal, SubscriptionTerminal)
    assert terminal.reason == "slow_consumer"


@pytest.mark.asyncio
async def test_broker_close_delivers_one_terminal_and_aclose_is_idempotent() -> None:
    broker = LiveStreamBroker(replay_capacity_per_scope=2, subscription_capacity=2)
    subscription = broker.subscribe(LiveSubscriptionRequest(scope="session", scope_id="session-1"))

    broker.close()
    broker.close()
    terminal = await _next(subscription)
    await subscription.aclose()
    await subscription.aclose()

    assert isinstance(terminal, SubscriptionTerminal)
    assert terminal.reason == "broker_closed"


@pytest.mark.asyncio
async def test_replay_scope_lru_bounds_history_and_reports_evicted_cursor() -> None:
    """连续完成 run 后回收旧 scope，旧游标重连明确收到 gap。"""
    broker = LiveStreamBroker(
        replay_capacity_per_scope=2,
        subscription_capacity=8,
        max_replay_scopes=4,
    )
    broker.publish(_run_event(1, run_id="run-old"))
    cursor = LiveCursor(
        stream_epoch=broker.current_epoch(),
        scope="run",
        scope_id="run-old",
        after_live_sequence=1,
    )
    for index in range(20):
        broker.publish(_run_event(1, run_id=f"run-{index}"))
        broker.publish(_run_event(2, run_id=f"run-{index}", kind=RunEventKind.RUN_TERMINAL))
        assert len(broker._rings) <= 4
        assert len(broker._sequences) <= 4

    subscription = broker.subscribe(
        LiveSubscriptionRequest(scope="run", scope_id="run-old", cursor=cursor)
    )
    gap = await _next(subscription)
    assert isinstance(gap, ReplayGap) and gap.reason == "unknown_cursor"
    await subscription.aclose()

    broker.publish(_run_event(2, run_id="run-old"))
    recreated = broker._rings[("run", "run-old")][-1].envelope
    assert recreated.live_sequence > cursor.after_live_sequence
    subscription = broker.subscribe(
        LiveSubscriptionRequest(scope="run", scope_id="run-old", cursor=cursor)
    )
    gap = await _next(subscription)
    assert isinstance(gap, ReplayGap) and gap.reason == "cursor_expired"
    await subscription.aclose()
    broker.close()
    assert broker._rings == {}
    assert broker._sequences == {}


@pytest.mark.asyncio
async def test_replay_eviction_preserves_unconsumed_active_scope_sequences() -> None:
    """回放被淘汰时保留活跃 scope 的 published high-water。"""
    broker = LiveStreamBroker(
        replay_capacity_per_scope=2,
        subscription_capacity=8,
        max_replay_scopes=1,
    )
    subscription = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-1"))
    for sequence in range(1, 4):
        broker.publish(_run_event(sequence))
        broker.publish(_run_event(1, run_id=f"other-{sequence}"))
    assert ("run", "run-1") not in broker._rings
    assert [item.live_sequence for item in [await _next(subscription) for _ in range(3)]] == [
        1,
        2,
        3,
    ]
    await subscription.aclose()
    assert ("run", "run-1") not in broker._sequences
    broker.close()


@pytest.mark.asyncio
async def test_subscribing_refreshes_replay_scope_lru_position() -> None:
    """最近访问的旧 run 在下一次淘汰时继续保留回放。"""
    broker = LiveStreamBroker(
        replay_capacity_per_scope=2,
        subscription_capacity=8,
        max_replay_scopes=3,
    )
    broker.publish(_run_event(1, run_id="run-first"))
    broker.publish(_run_event(1, run_id="run-second"))
    subscription = broker.subscribe(LiveSubscriptionRequest(scope="run", scope_id="run-first"))
    await subscription.aclose()
    broker.publish(_run_event(1, run_id="run-third"))

    assert ("run", "run-first") in broker._rings
    assert ("run", "run-second") not in broker._rings
    broker.close()


def test_projection_allowlists_tool_and_provider_payloads(tmp_path: Path) -> None:
    secret_path = (tmp_path / "secret.txt").resolve()
    tool_result = ToolResult(
        tool_use_id="tool-1",
        tool_name="read_secret",
        content=[TextBlock(text="safe preview")],
        is_error=True,
        error=ToolErrorInfo(
            code="READ_FAILED",
            message="safe message",
            retryable=False,
            details={"traceback": "secret traceback"},
        ),
        data={"secret": "raw data"},
        artifact=ToolArtifact(
            path=secret_path,
            mime_type="text/plain",
            size_bytes=12,
            preview="artifact preview",
        ),
        metadata={"token": "secret token"},
    )
    tool_fact = RuntimeStreamEvent(
        kind="tool.completed",
        run_id="run-1",
        session_id="session-1",
        activation_id="activation-1",
        step_index=0,
        tool_call_id="tool-1",
        tool_name="read_secret",
        tool_ordinal=1,
        tool_result=tool_result,
    )
    response = LLMResponse(
        provider="fake",
        id="response-1",
        model="model-1",
        content=[TextBlock(text="raw model output")],
        finish_reason="stop",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        metadata={"raw_provider": "secret provider field"},
    )
    model_fact = RuntimeStreamEvent(
        kind="model.event",
        run_id="run-1",
        session_id="session-1",
        activation_id="activation-1",
        step_index=0,
        model_event=ModelResponseCompleted(
            scope=ModelStreamScope(
                model_stream_id="stream-1",
                provider="fake",
                model="model-1",
                attempt=1,
            ),
            sequence=1,
            occurred_at=datetime.now(UTC),
            response=response,
            semantic_output_emitted=True,
        ),
    )

    tool_payload = project_live_fact(tool_fact)[0].payload
    model_payload = project_live_fact(model_fact)[0].payload
    serialized = json.dumps([tool_payload, model_payload], ensure_ascii=False)

    assert tool_payload["content"] == ["safe preview"]
    assert tool_payload["artifact"] == {
        "mime_type": "text/plain",
        "size_bytes": 12,
        "preview": "artifact preview",
    }
    assert model_payload["provider"] == "fake"
    assert model_payload["usage"] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }
    for secret in (
        str(secret_path),
        "raw data",
        "secret token",
        "secret traceback",
        "secret provider field",
        "raw model output",
    ):
        assert secret not in serialized
