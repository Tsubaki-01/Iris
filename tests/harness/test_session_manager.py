"""Per-session ``SessionManager`` admission 与事件语义测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from iris.exceptions import IrisRunConflictError, IrisRunStateError
from iris.harness import (
    AgentRunner,
    RunEvent,
    RunPhase,
    SessionEvent,
    SessionManager,
    SubmissionEvent,
)
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import (
    RunEventKind,
    RunStopReason,
)
from iris.message import LLMRequest, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import (
    BlockingProvider,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    """让后台 settlement callback 获得运行机会，直到条件成立。"""
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("后台状态未在预期事件循环轮次内完成")


async def _collect_until_closed(stream: AsyncIterator[SessionEvent]) -> list[SessionEvent]:
    """收集 manager close 前已排队的全部事件。"""
    return [event async for event in stream]


def _submissions(events: list[SessionEvent]) -> list[SubmissionEvent]:
    return [event for event in events if isinstance(event, SubmissionEvent)]


@pytest.mark.asyncio
async def test_idle_submit_returns_after_create_while_provider_is_running(tmp_path: Path) -> None:
    """Idle receipt 证明 create 已提交，但不等待 provider settlement。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-managed",
    )

    receipt = await manager.submit("开始")

    run = store.load_run(receipt.run_id)
    assert receipt.mode is None and receipt.state == "delivered"
    assert run is not None and run.phase is RunPhase.ACTIVE
    assert store.load_result(receipt.run_id) is None
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    provider.release.set()
    await _wait_until(lambda: store.load_result(receipt.run_id) is not None)
    await manager.close()


@pytest.mark.asyncio
async def test_steer_delivery_follows_durable_commit_and_preserves_fifo(tmp_path: Path) -> None:
    """每个 safe boundary 只 claim 队首，且 delivered 晚于 durable commit event。"""
    provider = BlockingProvider(text_response("一轮"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-steer",
    )
    stream = manager.events()
    current = await manager.submit("开始")
    first = await manager.submit("方向一", mode="steer")
    second = await manager.submit("方向二", mode="steer")

    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)
    await _wait_until(lambda: manager._current_run_id is None)
    await manager.close()
    events = await _collect_until_closed(stream)
    submissions = _submissions(events)

    assert [(event.submission_id, event.state) for event in submissions] == [
        (first.submission_id, "pending"),
        (second.submission_id, "pending"),
        (first.submission_id, "delivered"),
        (second.submission_id, "delivered"),
    ]
    for delivered in (first, second):
        delivered_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, SubmissionEvent)
            and event.submission_id == delivered.submission_id
            and event.state == "delivered"
        )
        assert any(
            isinstance(event, RunEvent) and event.kind is RunEventKind.MODEL_STEP_COMMITTED
            for event in events[:delivered_index]
        )
    assert [message.text for message in store.load_session("session-steer").messages] == [
        "开始",
        "一轮",
        "方向一",
        "一轮",
        "方向二",
        "一轮",
    ]


@pytest.mark.asyncio
async def test_steer_claim_uses_control_without_loading_run_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """锁内选择 steer 只需要当前控制事实，不读取完整 checkpoint/result。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    manager = SessionManager(runner, "session-control")
    current = await manager.submit("开始")
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    queued = await manager.submit("新方向", mode="steer")
    control = runner.get_run_control(current.run_id)

    def reject_full_snapshot(run_id: str) -> None:
        raise AssertionError("steer claim loaded a full RunSnapshot")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner, "get_run", reject_full_snapshot)
            claimed = await manager._steering.claim(current.run_id, control.current_activation_id)
        assert claimed is not None
        assert claimed.submission_id == queued.submission_id
        assert claimed.message.text == "新方向"
    finally:
        manager._steering.fail(queued.submission_id, "test_finished")
        provider.release.set()
        await manager.close(cancel_run=True)


@pytest.mark.asyncio
async def test_follow_up_waits_for_terminal_and_starts_one_run_at_a_time(tmp_path: Path) -> None:
    """Follow-up admission 不抢占 current run，并按 FIFO 串行 create。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-follow-up",
    )
    stream = manager.events()
    current = await manager.submit("第一轮")
    first = await manager.submit("第二轮", mode="follow_up")
    second = await manager.submit("第三轮", mode="follow_up")

    assert store.load_run(first.run_id) is None
    assert store.load_run(second.run_id) is None
    provider.release.set()
    await _wait_until(lambda: store.load_result(second.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert store.load_result(current.run_id) is not None
    assert store.load_result(first.run_id) is not None
    assert store.load_result(second.run_id) is not None
    assert [(event.submission_id, event.state) for event in submissions] == [
        (first.submission_id, "pending"),
        (second.submission_id, "pending"),
        (first.submission_id, "delivered"),
        (second.submission_id, "delivered"),
    ]


@pytest.mark.asyncio
async def test_follow_up_does_not_block_eligible_steer(tmp_path: Path) -> None:
    """两条 mode FIFO 独立，早到 follow-up 不阻塞 current-run steer。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-mixed",
    )
    stream = manager.events()
    current = await manager.submit("开始")
    follow_up = await manager.submit("以后", mode="follow_up")
    steer = await manager.submit("现在", mode="steer")

    provider.release.set()
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))
    delivered = [event.submission_id for event in submissions if event.state == "delivered"]

    assert delivered == [steer.submission_id, follow_up.submission_id]
    first_run_messages = [
        message.text
        for message in store.load_session("session-mixed").messages
        if message.text != "以后"
    ]
    assert "现在" in first_run_messages
    assert store.load_result(current.run_id) is not None


@pytest.mark.asyncio
async def test_waiting_run_accepts_steer_only_at_resume_boundary(tmp_path: Path) -> None:
    """HITL response 使用 exact resume，queued steer 等到 resumed activation boundary。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"})),
        text_response("已恢复"),
        text_response("已调整"),
    )
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider, registry=registry),
            store=store,
        ),
        "session-hitl",
    )
    stream = manager.events()
    current = await manager.submit("写入")
    await _wait_until(
        lambda: (
            (result := store.load_result(current.run_id)) is not None
            and result.run.phase is RunPhase.WAITING
        )
    )
    waiting = store.load_result(current.run_id)
    assert waiting is not None and waiting.pending_interaction is not None
    steer = await manager.submit("恢复后调整", mode="steer")

    result = await manager.resume(
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert [(event.submission_id, event.state) for event in submissions] == [
        (steer.submission_id, "pending"),
        (steer.submission_id, "delivered"),
    ]
    assert "恢复后调整" in [message.text for message in store.load_session("session-hitl").messages]


@pytest.mark.asyncio
async def test_active_interrupt_fails_steer_and_waits_terminal_before_follow_up(
    tmp_path: Path,
) -> None:
    """Cancellation request 不是 early follow-up eligibility。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-interrupt",
    )
    stream = manager.events()
    current = await manager.submit("第一轮")
    steer = await manager.submit("取消前调整", mode="steer")
    follow_up = await manager.submit("取消后继续", mode="follow_up")

    snapshot = await manager.interrupt(reason="用户停止")
    assert snapshot.run_id == current.run_id
    assert snapshot.phase is RunPhase.ACTIVE
    assert store.load_run(follow_up.run_id) is None
    await _wait_until(lambda: store.load_run(follow_up.run_id) is not None)
    provider.release.set()
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert any(
        event.submission_id == steer.submission_id
        and event.state == "failed"
        and event.reason == "target_cancelling"
        for event in submissions
    )
    assert any(
        event.submission_id == follow_up.submission_id and event.state == "delivered"
        for event in submissions
    )


@pytest.mark.asyncio
async def test_waiting_interrupt_settles_old_run_then_starts_follow_up(tmp_path: Path) -> None:
    """Waiting cancellation 同次 terminal 后才交付一条 follow-up，backfill 不重复事件。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(
                tmp_path,
                registry=registry,
                provider=StaticProvider(
                    tool_response(
                        ToolUseBlock(id="write-wait", name="write", input={"value": "x"})
                    ),
                    text_response("后续完成"),
                ),
            ),
            store=store,
        ),
        "session-waiting-interrupt",
    )
    stream = manager.events()
    current = await manager.submit("等待授权")
    await _wait_until(
        lambda: (
            (result := store.load_result(current.run_id)) is not None
            and result.run.phase is RunPhase.WAITING
        )
    )
    follow_up = await manager.submit("取消后继续", mode="follow_up")

    interrupted = await manager.interrupt(reason="放弃授权")
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    events = await _collect_until_closed(stream)

    assert interrupted.run_id == current.run_id
    assert interrupted.stop_reason is RunStopReason.CANCELLED
    run_events = [event for event in events if isinstance(event, RunEvent)]
    assert len({(event.run_id, event.sequence) for event in run_events}) == len(run_events)
    delivered_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, SubmissionEvent)
        and event.submission_id == follow_up.submission_id
        and event.state == "delivered"
    )
    assert any(
        isinstance(event, RunEvent)
        and event.run_id == follow_up.run_id
        and event.kind is RunEventKind.RUN_STARTED
        for event in events[:delivered_index]
    )


@pytest.mark.asyncio
async def test_close_fails_all_pending_then_ends_stream_without_cancelling_run(
    tmp_path: Path,
) -> None:
    """Close 只结束 façade ownership；pending 不 silent drop，durable run 不受影响。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-close",
    )
    stream = manager.events()
    current = await manager.submit("继续运行")
    steer = await manager.submit("调整", mode="steer")
    follow_up = await manager.submit("以后", mode="follow_up")

    await manager.close()
    await manager.close()
    events = await _collect_until_closed(stream)
    failures = [event for event in _submissions(events) if event.state == "failed"]

    assert [(event.submission_id, event.reason) for event in failures] == [
        (steer.submission_id, "session_closed"),
        (follow_up.submission_id, "session_closed"),
    ]
    active = store.load_run(current.run_id)
    assert active is not None and active.phase is RunPhase.ACTIVE
    assert active.cancellation_requested_at is None
    with pytest.raises(IrisRunStateError, match="closed"):
        await manager.submit("拒绝")
    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)


@pytest.mark.asyncio
async def test_events_has_one_consumer_and_new_manager_does_not_attach_lane(
    tmp_path: Path,
) -> None:
    """Event stream 与 current owner 均是 process-local，restart 不猜测 attach。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    owner = SessionManager(runner, "session-restart")
    first_stream = owner.events()
    with pytest.raises(IrisRunStateError, match="consumer"):
        owner.events()
    current = await owner.submit("占用 lane")
    restarted = SessionManager(runner, "session-restart")

    with pytest.raises(IrisRunConflictError, match="lane"):
        await restarted.submit("不得 attach")
    await restarted.close()
    assert [event async for event in restarted.events()] == []
    await owner.close()
    assert await _collect_until_closed(first_stream)
    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)


@pytest.mark.asyncio
async def test_stale_task_settlement_cannot_replace_new_current_owner(tmp_path: Path) -> None:
    """旧 task callback 必须由 exact task object 与 run id 双重 fence 忽略。"""

    class FirstThenBlockingProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        def estimate_input_tokens(self, request: LLMRequest) -> int:
            """为非计量测试返回固定输入估算。"""
            return 1

        async def complete(self, request: object):
            del request
            self.calls += 1
            if self.calls == 1:
                return text_response("第一轮完成")
            self.started.set()
            await self.release.wait()
            return text_response("第二轮完成")

    provider = FirstThenBlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-stale-task",
    )
    first = await manager.submit("第一轮")
    old_task = manager._current_task
    assert old_task is not None
    await _wait_until(lambda: store.load_result(first.run_id) is not None)
    await _wait_until(lambda: manager._current_run_id is None)
    second = await manager.submit("第二轮")
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    new_task = manager._current_task

    await manager._settle_managed_task(old_task, first.run_id, submission=None)

    assert manager._current_run_id == second.run_id
    assert manager._current_task is new_task
    provider.release.set()
    await _wait_until(lambda: store.load_result(second.run_id) is not None)
    await manager.close()
