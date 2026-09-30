"""Goal 最新快照合并与 mixed 终态交付顺序。"""

from datetime import UTC, datetime

import pytest

from iris.goal.models import GoalChanged
from iris.goal.service import GoalService
from iris.harness.session_manager import SubmissionEvent, _SessionEventBuffer
from iris.lifecycle import RunEvent, RunEventKind
from iris.store import InMemoryLifecycleStore


def _changed(objective: str) -> GoalChanged:
    """从领域读投影建立通知，不复制投影规则。"""
    service = GoalService(InMemoryLifecycleStore())
    service.create("session", objective)
    return GoalChanged(session_id="session", view=service.get_view("session"))


def _run_events(run_id: str, count: int = 2) -> list[RunEvent]:
    """返回可回读的模型事实及最后一条终态。"""
    return [
        RunEvent(
            run_id=run_id,
            session_id="session",
            sequence=index,
            kind=(
                RunEventKind.RUN_TERMINAL
                if index == count
                else RunEventKind.MODEL_STEP_COMMITTED
            ),
            occurred_at=datetime.now(UTC),
        )
        for index in range(1, count + 1)
    ]


def _buffer(rows: list[RunEvent], *, capacity: int = 2) -> _SessionEventBuffer:
    """保留真实 buffer 的有界回读，仅替换持久事件来源。"""
    def read(run_id: str, after_sequence: int = 0, *, limit: int | None = None) -> list[RunEvent]:
        selected = [row for row in rows if row.run_id == run_id and row.sequence > after_sequence]
        return selected if limit is None else selected[:limit]

    return _SessionEventBuffer(
        read,
        max_buffered_submission_events=2,
        max_tracked_durable_runs=capacity,
        on_tracker_released=lambda: None,
    )


@pytest.mark.asyncio
async def test_goal_notifications_use_one_slot_without_submission_reservations() -> None:
    """大量控制状态合并为最新快照，不消耗普通用户事件额度，关闭仍排空。"""
    buffer = _buffer([])
    for index in range(100):
        latest = _changed(f"目标 {index}")
        buffer.add_goal_changed(latest)
    assert buffer._goal_event is latest
    assert buffer._goal_barriers == {}
    assert buffer.buffered_submission_event_count == 0
    assert buffer.reserved_terminal_slots == 0
    assert buffer.can_reserve_submission_lifecycle()
    buffer.close()
    assert await buffer.next_event() is latest
    assert await buffer.next_event() is None


@pytest.mark.asyncio
async def test_coalesced_admission_keeps_both_terminal_barriers_and_submission_fifo() -> None:
    """新轮快照不能覆盖未交付旧轮终态，原用户 pending/terminal 顺序保留。"""
    first, second = _run_events("first"), _run_events("second")
    buffer = _buffer(first + second)
    assert buffer.try_register_run("first", after_sequence=0)
    buffer.observe_run_event(first[-1])
    buffer.add_goal_changed(_changed("第一轮结算"))
    pending = SubmissionEvent(
        submission_id="input", run_id="second", mode="follow_up", state="pending"
    )
    buffer.add_pending(pending)
    assert buffer.try_register_run("second", after_sequence=0)
    buffer.observe_run_event(second[0])
    buffer.add_goal_changed(_changed("第二轮准入"))
    buffer.observe_run_event(first[-1])
    buffer.observe_run_event(second[-1])
    latest = _changed("第二轮结算")
    buffer.add_goal_changed(latest)
    assert buffer._goal_barriers == {"first": 2, "second": 2}
    assert not buffer.try_register_run("third", after_sequence=0)
    assert await buffer.next_event() == first[0]
    assert await buffer.next_event() == first[1]
    assert buffer._goal_barriers == {"second": 2}
    assert await buffer.next_event() is pending
    assert await buffer.next_event() == second[0]
    assert await buffer.next_event() == second[1]
    assert buffer._goal_barriers == {}
    assert await buffer.next_event() is latest
    assert buffer.tracked_run_count == 0
    delivered = pending.model_copy(update={"state": "delivered"})
    buffer.add_terminal(delivered)
    assert await buffer.next_event() is delivered


@pytest.mark.asyncio
async def test_goal_terminal_barrier_survives_multiple_replay_batches() -> None:
    """终态位于第二个回读批次时，快照不能提前交付。"""
    rows = _run_events("long-run", 70)
    buffer = _buffer(rows, capacity=1)
    assert buffer.try_register_run("long-run", after_sequence=0)
    buffer.observe_run_event(rows[-1])
    buffer.add_goal_changed(_changed("已结算"))
    assert await buffer.next_event() == rows[0]
    latest = _changed("后续状态")
    buffer.add_goal_changed(latest)
    for expected in rows[1:]:
        assert await buffer.next_event() == expected
        assert buffer.durable_replay_batch_count <= 64
    assert await buffer.next_event() is latest
    assert buffer._goal_barriers == {}


@pytest.mark.asyncio
async def test_goal_control_without_terminal_is_not_held_by_active_run_progress() -> None:
    """无终态依赖的控制通知可以直接显示，不等待模型的在途事件。"""
    rows = _run_events("active")
    buffer = _buffer(rows)
    assert buffer.try_register_run("active", after_sequence=0)
    buffer.observe_run_event(rows[0])
    latest = _changed("用户修改")
    buffer.add_goal_changed(latest)
    assert await buffer.next_event() is latest
    assert await buffer.next_event() == rows[0]
