"""AgentRunner committed-event observer delivery 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunEvent, RunEventKind, RunStopReason
from iris.store import InMemoryLifecycleStore

from .fakes import build_runtime


class RecordingObserver:
    """按收到顺序记录 event 的 observer。"""

    def __init__(self, name: str, target: list[tuple[str, int]]) -> None:
        self.name = name
        self.target = target

    async def on_event(self, event: RunEvent) -> None:
        """记录 observer 名称和 durable sequence。"""
        self.target.append((self.name, event.sequence))


@pytest.mark.asyncio
async def test_observers_receive_committed_events_in_event_then_registration_order(
    tmp_path: Path,
) -> None:
    """事件按 sequence，且同一事件按 observer 注册顺序投递。"""
    deliveries: list[tuple[str, int]] = []
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=store,
        observers=[
            RecordingObserver("first", deliveries),
            RecordingObserver("second", deliveries),
        ],
    )

    await runner.start(AgentRunRequest(input="观察", run_id="run-observed"))

    sequences = [event.sequence for event in store.list_events("run-observed")]
    assert deliveries == [
        (observer, sequence) for sequence in sequences for observer in ("first", "second")
    ]


@pytest.mark.asyncio
async def test_observer_only_sees_events_already_queryable_from_store(
    tmp_path: Path,
) -> None:
    """observer callback 不能先于权威 store commit。"""
    store = InMemoryLifecycleStore()
    checked: list[int] = []

    class CommitCheckingObserver:
        async def on_event(self, event: RunEvent) -> None:
            persisted = store.list_events(event.run_id)
            assert event in persisted
            assert store.load_run(event.run_id) is not None
            checked.append(event.sequence)

    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=store,
        observers=[CommitCheckingObserver()],
    )

    await runner.start(AgentRunRequest(input="观察", run_id="run-committed"))

    assert checked == [event.sequence for event in store.list_events("run-committed")]


@pytest.mark.asyncio
async def test_observer_failure_does_not_rewrite_outcome_and_events_are_backfillable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """best-effort 投递失败后，结果和 durable event log 保持不变。"""
    delivered: list[tuple[str, int]] = []
    store = InMemoryLifecycleStore()

    class FailingObserver:
        async def on_event(self, event: RunEvent) -> None:
            raise RuntimeError(f"observer failed at {event.sequence}")

    runner = AgentRunner(
        runtime=build_runtime(tmp_path),
        store=store,
        observers=[FailingObserver(), RecordingObserver("healthy", delivered)],
    )

    result = await runner.start(AgentRunRequest(input="观察", run_id="run-observer-failure"))

    durable_events = store.list_events("run-observer-failure")
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert delivered == [("healthy", event.sequence) for event in durable_events]
    assert runner.list_events("run-observer-failure", after_sequence=2) == durable_events[2:]
    assert durable_events[-1].kind is RunEventKind.RUN_TERMINAL
    assert "observer 处理失败" in caplog.text
