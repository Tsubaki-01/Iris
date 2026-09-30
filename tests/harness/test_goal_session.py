"""Goal 与单会话准入、用户优先和控制的集成契约。"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from iris.exceptions import IrisGoalStateError
from iris.goal.models import GoalChanged, GoalView
from iris.harness import AgentRunner, SessionManager
from iris.lifecycle import RunPhase
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore

from .fakes import BlockingProvider, StaticProvider, text_response, tool_response
from .test_goal_execution import goal_config


async def goal_state(events: AsyncIterator[object], status: str) -> GoalView:
    """从真实混合事件流等待指定目标状态。"""
    async with asyncio.timeout(5):
        async for event in events:
            if isinstance(event, GoalChanged) and event.view.goal is not None:
                if event.view.goal.status == status:
                    return event.view
    raise AssertionError("事件流提前结束")


@pytest.mark.asyncio
async def test_goal_continues_two_rounds_then_committed_report_completes(tmp_path: Path) -> None:
    store = InMemoryLifecycleStore()

    class Provider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 2:
                goal = store.get_current_goal("s")
                return tool_response(
                    ToolUseBlock(
                        id="complete",
                        name="report_goal",
                        input={
                            "goal_id": goal.goal_id,
                            "revision": goal.revision,
                            "decision": "complete",
                            "reason": "已检查两轮结果",
                        },
                    )
                )
            return text_response()

    provider = Provider()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(runner, "s")
    try:
        created = await manager.goal.create("两轮完成", max_rounds=2)
        assert created.disposition == "scheduled"
        view = await goal_state(manager.events(), "completed")
        assert view.goal.rounds_started == 2 and not view.armed
        assert len(provider.requests) == 3
        assert len(store._runs) == 2
        assert all(run.phase is RunPhase.TERMINAL for run in store._runs.values())
        assert store.list_unsettled_goal_runs("s") == ()
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_round_limit_stops_and_resume_does_not_reset_spent_rounds(tmp_path: Path) -> None:
    provider = StaticProvider(text_response())
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("没有申报", max_rounds=1)
        view = await goal_state(manager.events(), "paused")
        assert view.goal.reason.code == "round_limit"
        resumed = await manager.goal.resume()
        assert resumed.disposition == "stopped"
        assert resumed.view.goal.rounds_started == 1
        assert len(provider.requests) == 1
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_queued_user_work_precedes_goal_without_spending_goal_rounds(tmp_path: Path) -> None:
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(runner, "s")
    try:
        first = await manager.submit("普通第一轮")
        await provider.started.wait()
        await manager.goal.create("后台目标", max_rounds=1)
        follow_up = await manager.submit("排队用户工作", mode="follow_up")
        assert (await manager.goal.get()).goal.rounds_started == 0
        provider.release.set()
        view = await goal_state(manager.events(), "paused")
        run_ids = list(store._runs)
        assert run_ids[:2] == [first.run_id, follow_up.run_id]
        assert view.goal.rounds_started == 1 and len(run_ids) == 3
        assert store.get_goal_run(follow_up.run_id) is None
    finally:
        provider.release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_pause_during_prepare_allows_user_and_preserves_shared_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StaticProvider(text_response())
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    runner._prepared = False
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def prepare(environment: object) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()

    monkeypatch.setattr(type(runner.runtime.environment), "aprepare", prepare)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("准备中可暂停")
        async with asyncio.timeout(5):
            await entered.wait()
            stopped = await manager.goal.pause(reason="用户先处理")
            assert stopped.view.goal.rounds_started == 0
            user = asyncio.create_task(manager.submit("优先用户问题"))
            release.set()
            receipt = await user
        assert calls == 1
        assert runner._goal_service.store.get_goal_run(receipt.run_id) is None
        assert (await manager.goal.get()).goal.status == "paused"
    finally:
        release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_idle_goal_interrupt_and_manager_attachment_lifecycle(tmp_path: Path) -> None:
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=StaticProvider())
    manager = SessionManager(runner, "s")
    with pytest.raises(IrisGoalStateError):
        SessionManager(runner, "s")
    await manager.goal.create("尚未准入")
    assert await manager.interrupt() is None
    assert (await manager.goal.get()).goal.status == "paused"
    await manager.close()
    second = SessionManager(runner, "s")
    assert not (await second.goal.get()).armed
    await second.close()
    await runner.aclose()


@pytest.mark.asyncio
async def test_user_submit_preempts_goal_prepare_then_goal_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StaticProvider(text_response("用户完成"), text_response("目标本轮结束"))
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    runner._prepared = False
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(environment: object) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(type(runner.runtime.environment), "aprepare", prepare)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("用户之后继续", max_rounds=1)
        async with asyncio.timeout(5):
            await entered.wait()
            user = asyncio.create_task(manager.submit("先答用户"))
            await asyncio.sleep(0)
            release.set()
            receipt = await user
        view = await goal_state(manager.events(), "paused")
        assert list(store._runs)[0] == receipt.run_id
        assert store.get_goal_run(receipt.run_id) is None
        assert view.goal.rounds_started == 1
        assert len(provider.requests) == 2
    finally:
        release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_goal_run_accepts_real_steer_and_invalidates_early_report(tmp_path: Path) -> None:
    store = InMemoryLifecycleStore()
    entered, release = asyncio.Event(), asyncio.Event()

    class Provider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                goal = store.get_current_goal("s")
                entered.set()
                await release.wait()
                return tool_response(
                    ToolUseBlock(
                        id="early",
                        name="report_goal",
                        input={
                            "goal_id": goal.goal_id,
                            "revision": goal.revision,
                            "decision": "complete",
                            "reason": "原请求已做完",
                        },
                    )
                )
            return text_response()

    runner = AgentRunner.from_config(goal_config(tmp_path), provider=Provider(), store=store)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("接收后到用户要求", max_rounds=1)
        async with asyncio.timeout(5):
            await entered.wait()
        receipt = await manager.submit("补充验收要求", mode="auto")
        release.set()
        view = await goal_state(manager.events(), "paused")
        assert receipt.mode == "steer"
        assert len(store._runs) == 1 and view.goal.rounds_started == 1
        assert any(message.text == "补充验收要求" for message in store.load_session("s").messages)
        assert store.get_goal_run(receipt.run_id).applied_report_call_id is None
    finally:
        release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()
