"""SessionManager、observer 与 settlement 的可重复性能观测。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

import pytest

from iris.harness import AgentRunner, AgentRunRequest, RunEvent, RunResult, SessionManager
from iris.lifecycle import RunEventKind
from iris.store import InMemoryLifecycleStore
from tests.harness.fakes import build_runtime

_EVENT_COUNT = 1_000
_OBSERVER_COUNT = 3
_OBSERVER_EVENT_COUNT = 5
_OBSERVER_DELAY_S = 0.02


def _events(count: int, *, run_id: str) -> list[RunEvent]:
    return [
        RunEvent(
            run_id=run_id,
            session_id="session-performance",
            sequence=index + 1,
            kind=(
                RunEventKind.RUN_TERMINAL
                if index == count - 1
                else RunEventKind.MODEL_STEP_COMMITTED
            ),
            occurred_at=datetime.now(UTC),
        )
        for index in range(count)
    ]


class _ReplayRunner:
    def __init__(self, events: list[RunEvent]) -> None:
        self.events = events
        self.list_calls = 0
        self.returned_rows = 0
        self.max_returned_batch = 0

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]:
        self.list_calls += 1
        events = [
            event
            for event in self.events
            if event.run_id == run_id and event.sequence > after_sequence
        ]
        selected = events if limit is None else events[:limit]
        self.returned_rows += len(selected)
        self.max_returned_batch = max(self.max_returned_batch, len(selected))
        return selected


@pytest.mark.asyncio
@pytest.mark.performance_timing
async def test_p5_manager_durable_burst_observation(
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    del require_performance_timing
    events = _events(_EVENT_COUNT, run_id="run-manager-performance")
    replay = _ReplayRunner(events)
    manager = SessionManager(
        cast(AgentRunner, replay),
        "session-manager-performance",
        max_tracked_durable_runs=1,
    )
    stream = manager.events()
    manager._event_buffer.register_run("run-manager-performance", after_sequence=0)

    started = perf_counter()
    for event in reversed(events):
        manager._relay_run_event(event)
        manager._relay_run_event(event)
    await manager.close()
    delivered: list[RunEvent] = []
    replay_batch_peak = 0
    async for event in stream:
        assert isinstance(event, RunEvent)
        delivered.append(event)
        replay_batch_peak = max(
            replay_batch_peak,
            manager._event_buffer.durable_replay_batch_count,
        )
    elapsed_ms = (perf_counter() - started) * 1000

    assert [event.sequence for event in delivered] == list(range(1, _EVENT_COUNT + 1))
    assert manager._event_buffer.tracked_run_count == 0
    assert replay_batch_peak <= 64
    assert replay.max_returned_batch <= 64
    assert replay.returned_rows == _EVENT_COUNT
    record_observation(
        scenario="p5_manager_durable_burst",
        perf_ids=("PERF-08",),
        fixture={"events": _EVENT_COUNT, "duplicate_relays_per_event": 2},
        samples_ms=(elapsed_ms,),
        counters={
            "active_tracker_peak": 1,
            "durable_replay_batch_peak": replay_batch_peak,
            "store_returned_rows": replay.returned_rows,
            "store_returned_batch_peak": replay.max_returned_batch,
            "store_list_calls": replay.list_calls,
            "retained_tracker_count": manager._event_buffer.tracked_run_count,
        },
    )


@pytest.mark.asyncio
@pytest.mark.performance_timing
async def test_p5_observer_lane_observation(
    tmp_path: Path,
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    del require_performance_timing
    active = 0
    maximum_active = 0
    orders: list[list[int]] = [[] for _ in range(_OBSERVER_COUNT)]

    class SlowObserver:
        def __init__(self, index: int) -> None:
            self.index = index

        async def on_event(self, event: RunEvent) -> None:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                orders[self.index].append(event.sequence)
                await asyncio.sleep(_OBSERVER_DELAY_S)
            finally:
                active -= 1

    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=InMemoryLifecycleStore(),
        observers=tuple(SlowObserver(index) for index in range(_OBSERVER_COUNT)),
    )
    events = _events(_OBSERVER_EVENT_COUNT, run_id="run-observer-performance")
    await runner._deliver_events(events)
    orders = [[] for _ in range(_OBSERVER_COUNT)]
    maximum_active = 0
    samples: list[float] = []
    for _ in range(5):
        orders = [[] for _ in range(_OBSERVER_COUNT)]
        started = perf_counter()
        await runner._deliver_events(events)
        samples.append((perf_counter() - started) * 1000)
        assert all(order == list(range(1, _OBSERVER_EVENT_COUNT + 1)) for order in orders)

    assert maximum_active == _OBSERVER_COUNT
    record_observation(
        scenario="p5_observer_parallel_lanes",
        perf_ids=("PERF-10",),
        fixture={
            "events": _OBSERVER_EVENT_COUNT,
            "observers": _OBSERVER_COUNT,
            "delay_ms": int(_OBSERVER_DELAY_S * 1000),
        },
        samples_ms=tuple(samples),
        counters={
            "observer_lane_peak": maximum_active,
            "observer_calls_per_sample": _OBSERVER_COUNT * _OBSERVER_EVENT_COUNT,
        },
    )


@pytest.mark.asyncio
@pytest.mark.performance_timing
async def test_p5_settlement_polling_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    del require_performance_timing
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    completed = await runner.start(
        AgentRunRequest(input="完成", run_id="run-settlement-performance")
    )
    original_load_result = store.load_result
    load_result_calls = 0
    sleep_intervals: list[float] = []

    def delayed_load_result(run_id: str) -> RunResult | None:
        nonlocal load_result_calls
        load_result_calls += 1
        if load_result_calls < 3:
            return None
        return original_load_result(run_id)

    async def record_sleep(interval: float) -> None:
        sleep_intervals.append(interval)

    monkeypatch.setattr(store, "load_result", delayed_load_result)
    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    started = perf_counter()
    observed = await runner._observe_settlement(
        completed.run.run_id,
        settlement_timeout=None,
    )
    elapsed_ms = (perf_counter() - started) * 1000

    assert observed == completed
    assert sleep_intervals == [0.05, 0.05]
    record_observation(
        scenario="p5_settlement_polling_unchanged",
        perf_ids=("PERF-13",),
        fixture={"missing_reads_before_terminal": 2},
        samples_ms=(elapsed_ms,),
        counters={
            "load_result_reads": load_result_calls,
            "poll_sleeps": len(sleep_intervals),
            "poll_interval_ms": 50,
        },
    )
