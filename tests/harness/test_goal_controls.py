"""Goal 控制、用户取消和两种观察模式的核心集成时序。"""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from iris.exceptions import IrisGoalPersistenceError, IrisGoalStateError
from iris.goal.models import GoalChanged, GoalSnapshot
from iris.goal.store import GoalUpdate, PauseGoal
from iris.harness import AgentRunner, SessionManager
from iris.lifecycle import RunPhase
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore

from .fakes import (
    BlockingProvider,
    FailingPublisher,
    RecordingPublisher,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)
from .test_goal_execution import goal_config
from .test_goal_session import goal_state


async def _until(predicate: Callable[[], bool]) -> None:
    """等待可观察的确定性状态，不以固定睡眠猜测执行速度。"""
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0)


class _TwoRoundProvider(StaticProvider):
    """第一轮正常结束，第二轮申报完成后正常收尾。"""

    def __init__(self, store: InMemoryLifecycleStore) -> None:
        super().__init__()
        self.store = store

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """从真实当前目标取申报身份。"""
        self.requests.append(request)
        if len(self.requests) == 2:
            goal = self.store.get_current_goal("s")
            assert goal is not None
            return tool_response(
                ToolUseBlock(
                    id="complete", name="report_goal",
                    input={
                        "goal_id": goal.goal_id, "revision": goal.revision,
                        "decision": "complete", "reason": "两轮工作完成",
                    },
                )
            )
        return text_response()


@pytest.mark.asyncio
async def test_clear_then_create_keeps_old_run_bound_to_old_goal(tmp_path: Path) -> None:
    """旧执行返回的完成申报无法完成替换目标，旧上下文也不能接收新正文。"""
    store = InMemoryLifecycleStore()
    started, release = asyncio.Event(), asyncio.Event()

    class Provider(StaticProvider):
        """把旧模型响应延迟到用户已替换目标之后。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """旧申报固定旧 ID/revision，其余请求正常结束。"""
            self.requests.append(request)
            if len(self.requests) == 1:
                old = store.get_current_goal("s")
                assert old is not None
                started.set()
                await release.wait()
                return tool_response(
                    ToolUseBlock(
                        id="old-report", name="report_goal",
                        input={
                            "goal_id": old.goal_id, "revision": old.revision,
                            "decision": "complete", "reason": "旧工作完成",
                        },
                    )
                )
            return text_response()

    provider = Provider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(runner, "s")
    try:
        first = await manager.goal.create("旧目标", max_rounds=1)
        await asyncio.wait_for(started.wait(), 5)
        old_run_id = manager._current_run_id
        await manager.goal.clear()
        second = await manager.goal.create("新目标不可注入旧轮", max_rounds=1)
        assert second.view.goal.goal_id != first.view.goal.goal_id
        assert second.view.run_goal_id == first.view.goal.goal_id
        assert second.view.run.cancellation_requested_at is None
        release.set()
        view = await goal_state(manager.events(), "paused")
        assert view.goal.goal_id == second.view.goal.goal_id
        assert view.goal.rounds_started == 1
        assert view.goal.reason.code == "round_limit"
        assert store.get_goal(first.view.goal.goal_id).status == "active"
        old_binding = store.get_goal_run(old_run_id)
        assert old_binding.goal_id == first.view.goal.goal_id and old_binding.settled_at is not None
        assert runner.list_tool_calls(old_run_id)[0].result.is_error
        assert "新目标不可注入旧轮" not in "\n".join(
            message.text for message in provider.requests[1].messages
        )
        assert "新目标不可注入旧轮" in "\n".join(
            message.text for message in provider.requests[2].messages
        )
    finally:
        release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["pause", "complete"])
async def test_goal_stop_control_preserves_current_run(tmp_path: Path, operation: str) -> None:
    """持久目标停止只阻止后续轮次，当前已准入执行仍正常完成。"""
    provider = BlockingProvider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("当前轮可以收尾")
        await asyncio.wait_for(provider.started.wait(), 5)
        run_id = manager._current_run_id
        stopped = (
            await manager.goal.pause(reason="稍后继续")
            if operation == "pause"
            else await manager.goal.complete(reason="用户确认完成")
        )
        assert not stopped.view.armed
        assert stopped.view.goal.status == ("paused" if operation == "pause" else "completed")
        assert runner.get_run(run_id).phase is RunPhase.ACTIVE
        assert runner.get_run(run_id).cancellation_requested_at is None
        provider.release.set()
        await _until(lambda: manager._current_run_id is None)
        assert runner.get_result(run_id).run.stop_reason == "completed"
        assert (await manager.goal.get()).goal.status == stopped.view.goal.status
        assert len(provider.requests) == 1
    finally:
        provider.release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_idle_goal_interrupt_and_completed_goal_do_not_block_chat_cancel(
    tmp_path: Path,
) -> None:
    """空闲意图取消不伪造 Run，已完成目标也不妨碍普通聊天取消。"""
    provider = BlockingProvider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("先保存目标")
        assert await manager.interrupt() is None
        assert manager._current_run_id is None
        assert not provider.requests
        await manager.goal.complete(reason="用户已完成")
        completed = (await manager.goal.get()).goal
        receipt = await manager.submit("普通聊天")
        await asyncio.wait_for(provider.started.wait(), 5)
        cancelled = await manager.interrupt(reason="结束普通聊天")
        assert cancelled.run_id == receipt.run_id
        assert cancelled.cancellation_requested_at is not None
        assert (await manager.goal.get()).goal == completed
        await _until(lambda: runner.get_run(receipt.run_id).phase is RunPhase.TERMINAL)
        assert runner.get_result(receipt.run_id).run.stop_reason == "cancelled"
    finally:
        provider.release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_goal_pause_write_failure_still_cancels_actual_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目标持久化失败不能吞掉用户对当前 Run 的取消要求。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("正在执行")
        await asyncio.wait_for(provider.started.wait(), 5)
        run_id = manager._current_run_id
        original = store.update_goal

        def fail_pause(command: GoalUpdate) -> GoalSnapshot | None:
            """只使目标暂停写入失败，不影响独立 run 取消事务。"""
            if isinstance(command, PauseGoal):
                raise IrisGoalPersistenceError("目标暂停写入失败")
            return original(command)

        monkeypatch.setattr(store, "update_goal", fail_pause)
        with pytest.raises(IrisGoalPersistenceError, match="目标暂停写入失败") as caught:
            await manager.interrupt(reason="仍需取消当前执行")
        assert runner.get_run(run_id).cancellation_requested_at is not None
        assert any(run_id in note for note in caught.value.__notes__)
        assert not (await manager.goal.get()).armed
        await _until(lambda: runner.get_run(run_id).phase is RunPhase.TERMINAL)
        assert runner.get_result(run_id).run.stop_reason == "cancelled"
        assert len(provider.requests) == 1
    finally:
        provider.release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_disabled_has_no_attachment_and_new_manager_get_does_not_reconcile(
    tmp_path: Path,
) -> None:
    """关闭开关保留原 manager 构造；启用后的 get 只展示待结算，不暗中恢复。"""
    disabled = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    first, second = SessionManager(disabled, "ordinary"), SessionManager(disabled, "ordinary")
    assert first.goal is None and second.goal is None
    await first.close()
    await second.close()
    await disabled.aclose()

    provider = StaticProvider(text_response())
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    service = runner._goal_service
    goal = service.create("s", "终态等待显式结算", max_rounds=2)
    result = await runner._start_goal_managed(goal.ref, run_id="pending")
    assert result.run.phase is RunPhase.TERMINAL
    manager = SessionManager(runner, "s")
    try:
        with pytest.raises(IrisGoalStateError):
            SessionManager(runner, "s")
        before = service.get(goal.goal_id)
        view = await manager.goal.get()
        assert view.settlement_pending and not view.armed
        assert view.run is None
        assert service.get(goal.goal_id) == before
        assert service.store.get_goal_run("pending").settled_at is None
        assert len(provider.requests) == 1
        await manager.close()
        replacement = SessionManager(runner, "s")
        try:
            view = await replacement.goal.get()
            assert view.settlement_pending and not view.armed
            assert service.store.get_goal_run("pending").settled_at is None
        finally:
            await replacement.close()
    finally:
        await manager.close()
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_fails", [False, True])
async def test_broker_only_goal_runs_without_event_consumer(
    tmp_path: Path, publish_fails: bool
) -> None:
    """通知成功或失败都不决定目标是否能继续，broker-only 不占本地 tracker。"""
    store = InMemoryLifecycleStore()
    provider = _TwoRoundProvider(store)
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    publisher = FailingPublisher() if publish_fails else RecordingPublisher()
    manager = SessionManager(
        runner, "s", observation_mode="broker_only", submission_publisher=publisher,
        max_tracked_durable_runs=1,
    )
    try:
        await manager.goal.create("两轮后完成", max_rounds=2)
        await _until(lambda: store.get_current_goal("s").status == "completed")
        assert manager._event_buffer is None
        assert store.get_current_goal("s").rounds_started == 2
        assert len(provider.requests) == 3
        assert any(
            isinstance(fact, GoalChanged) and fact.view.goal.status == "completed"
            for fact in publisher.facts
        )
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_mixed_tracker_capacity_waits_then_continues_after_terminal_delivery(
    tmp_path: Path,
) -> None:
    """单 tracker 下不提前创建第二轮，消费者交付终态才唤醒后续准入。"""
    store = InMemoryLifecycleStore()
    provider = _TwoRoundProvider(store)
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(runner, "s", max_tracked_durable_runs=1)
    try:
        await manager.goal.create("两轮后完成", max_rounds=2)
        await _until(
            lambda: len(provider.requests) == 1
            and manager._current_run_id is None
            and not store.list_unsettled_goal_runs("s")
            and not manager._managed_tasks
            and not manager._goal_control._operations
            and manager._tracker_reconcile_task is None
        )
        view = await manager.goal.get()
        assert view.goal.rounds_started == 1 and view.armed
        assert manager._event_buffer.tracked_run_count == 1
        assert manager._goal_control.driver.intent is None
        completed = await goal_state(manager.events(), "completed")
        assert completed.goal.rounds_started == 2
        assert len(provider.requests) == 3
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()
