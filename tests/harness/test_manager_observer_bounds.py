"""SessionManager 有界状态与 runner observer 投递语义测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from iris.exceptions import IrisRunStateError
from iris.harness import (
    AgentRunner,
    AgentRunRequest,
    RunEvent,
    RunPhase,
    SessionManager,
    SubmissionEvent,
)
from iris.lifecycle import RunEventKind, RunStopReason
from iris.store import InMemoryLifecycleStore

from .fakes import BlockingProvider, build_runtime


def _event(sequence: int, *, run_id: str = "run-replay") -> RunEvent:
    return RunEvent(
        run_id=run_id,
        session_id="session-bounds",
        sequence=sequence,
        kind=RunEventKind.MODEL_STEP_COMMITTED,
        occurred_at=datetime.now(UTC),
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("后台状态未在预期事件循环轮次内完成")


class _ReplayRunner:
    def __init__(self, events: list[RunEvent]) -> None:
        self.events = events
        self.on_first_list: Callable[[], None] | None = None
        self.list_calls = 0
        self.requested_limits: list[int | None] = []
        self.returned_rows = 0

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]:
        self.list_calls += 1
        self.requested_limits.append(limit)
        if self.on_first_list is not None:
            callback, self.on_first_list = self.on_first_list, None
            callback()
        events = [
            event
            for event in self.events
            if event.run_id == run_id and event.sequence > after_sequence
        ]
        selected = events if limit is None else events[:limit]
        self.returned_rows += len(selected)
        return selected


@pytest.mark.asyncio
async def test_busy_admission_rejects_before_queue_or_event_side_effects(tmp_path: Path) -> None:
    provider = BlockingProvider()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-admission-bounds",
        max_pending_steer=1,
        max_pending_follow_up=1,
        max_buffered_submission_events=4,
    )
    stream = manager.events()
    await manager.submit("占用")
    steer = await manager.submit("steer-1", mode="steer")
    follow_up = await manager.submit("follow-up-1", mode="follow_up")
    before = (
        manager._pending.steer_count,
        manager._pending.follow_up_count,
        manager._event_buffer.buffered_submission_event_count,
        manager._event_buffer.reserved_terminal_slots,
    )

    with pytest.raises(IrisRunStateError, match="steer.*容量"):
        await manager.submit("steer-rejected", mode="steer")
    with pytest.raises(IrisRunStateError, match="follow_up.*容量"):
        await manager.submit("follow-up-rejected", mode="follow_up")

    assert (
        (
            manager._pending.steer_count,
            manager._pending.follow_up_count,
            manager._event_buffer.buffered_submission_event_count,
            manager._event_buffer.reserved_terminal_slots,
        )
        == before
        == (1, 1, 2, 2)
    )
    await manager.close()
    submissions = [event async for event in stream if isinstance(event, SubmissionEvent)]
    assert [event.submission_id for event in submissions if event.state == "pending"] == [
        steer.submission_id,
        follow_up.submission_id,
    ]
    assert [event.submission_id for event in submissions if event.state == "failed"] == [
        steer.submission_id,
        follow_up.submission_id,
    ]
    provider.release.set()


@pytest.mark.asyncio
async def test_submission_event_capacity_reserves_terminal_before_accepting(tmp_path: Path) -> None:
    provider = BlockingProvider()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-event-capacity",
        max_pending_steer=2,
        max_buffered_submission_events=2,
    )
    stream = manager.events()
    await manager.submit("占用")
    accepted = await manager.submit("accepted", mode="steer")

    with pytest.raises(IrisRunStateError, match="submission event.*容量"):
        await manager.submit("rejected", mode="steer")

    while True:
        first = await anext(stream)
        if isinstance(first, SubmissionEvent):
            break
    assert first.submission_id == accepted.submission_id and first.state == "pending"
    assert manager._event_buffer.buffered_submission_event_count == 0
    assert manager._event_buffer.reserved_terminal_slots == 1

    await manager.close()
    remaining = [event async for event in stream]
    terminal = [
        event
        for event in remaining
        if isinstance(event, SubmissionEvent) and event.submission_id == accepted.submission_id
    ]
    assert [(event.state, event.reason) for event in terminal] == [("failed", "session_closed")]
    assert manager._event_buffer.buffered_submission_event_count == 0
    assert manager._event_buffer.reserved_terminal_slots == 0
    provider.release.set()


@pytest.mark.asyncio
async def test_no_consumer_keeps_all_manager_containers_within_small_limits(tmp_path: Path) -> None:
    provider = BlockingProvider()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-small-limits",
        max_pending_steer=2,
        max_pending_follow_up=2,
        max_buffered_submission_events=8,
        max_tracked_durable_runs=1,
    )
    await manager.submit("占用")
    for index in range(2):
        await manager.submit(f"steer-{index}", mode="steer")
        await manager.submit(f"follow-up-{index}", mode="follow_up")

    assert manager._pending.steer_count == 2
    assert manager._pending.follow_up_count == 2
    assert (
        manager._event_buffer.buffered_submission_event_count
        + manager._event_buffer.reserved_terminal_slots
        == 8
    )
    assert manager._event_buffer.tracked_run_count == 1
    await manager.close()
    provider.release.set()


@pytest.mark.asyncio
async def test_durable_replay_uses_registration_baseline_and_catches_concurrent_relay() -> None:
    durable = [_event(sequence) for sequence in range(1, 6)]
    replay = _ReplayRunner(durable[:4])
    manager = SessionManager(
        cast(AgentRunner, replay),
        "session-replay-baseline",
        max_tracked_durable_runs=1,
    )
    stream = manager.events()
    assert manager._event_buffer.try_register_run("run-replay", after_sequence=3)
    manager._relay_run_event(durable[3])

    def append_during_read() -> None:
        replay.events.append(durable[4])
        manager._relay_run_event(durable[4])

    replay.on_first_list = append_during_read
    first = await anext(stream)
    await manager.close()
    delivered = [first, *[event async for event in stream if isinstance(event, RunEvent)]]

    assert [event.sequence for event in delivered] == [4, 5]
    assert replay.list_calls >= 2
    assert all(limit is not None and limit <= 64 for limit in replay.requested_limits)


@pytest.mark.asyncio
async def test_observer_lanes_run_in_parallel_and_preserve_each_observer_order(
    tmp_path: Path,
) -> None:
    active = 0
    maximum_active = 0
    orders: dict[str, list[int]] = {"left": [], "right": []}

    class RecordingObserver:
        def __init__(self, name: str) -> None:
            self.name = name

        async def on_event(self, event: RunEvent) -> None:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                orders[self.name].append(event.sequence)
                await asyncio.sleep(0)
            finally:
                active -= 1

    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=InMemoryLifecycleStore(),
        observers=(RecordingObserver("left"), RecordingObserver("right")),
    )

    await runner._deliver_events([_event(3), _event(1), _event(2), _event(2)])

    assert maximum_active == 2
    assert orders == {"left": [1, 2, 3], "right": [1, 2, 3]}


@pytest.mark.asyncio
async def test_observer_timeout_and_exception_do_not_block_other_lane_or_result(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    observed: list[int] = []

    class TimeoutObserver:
        async def on_event(self, event: RunEvent) -> None:
            del event
            await asyncio.Event().wait()

    class FailingObserver:
        async def on_event(self, event: RunEvent) -> None:
            raise RuntimeError(f"observer failed at {event.sequence}")

    class RecordingObserver:
        async def on_event(self, event: RunEvent) -> None:
            observed.append(event.sequence)

    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=store,
        observers=(TimeoutObserver(), FailingObserver(), RecordingObserver()),
        observer_event_timeout_s=0.01,
    )

    result = await asyncio.wait_for(runner.start(AgentRunRequest(input="完成")), timeout=1)
    durable = store.list_events(result.run.run_id)

    assert result.run.phase is RunPhase.TERMINAL
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert observed == [event.sequence for event in durable]
    assert "observer event 超时" in caplog.text
    assert "observer 处理失败" in caplog.text


@pytest.mark.asyncio
async def test_observer_cancellation_propagates(tmp_path: Path) -> None:
    class CancellingObserver:
        async def on_event(self, event: RunEvent) -> None:
            del event
            raise asyncio.CancelledError

    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=InMemoryLifecycleStore(),
        observers=(CancellingObserver(),),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner._deliver_events([_event(1)])
