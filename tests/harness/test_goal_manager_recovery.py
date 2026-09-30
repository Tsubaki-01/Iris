"""Goal Manager 在 SQLite 重启、原 Run 附着和终态恢复中的集成契约。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisRunConflictError, IrisRunPersistenceError
from iris.goal import GoalChanged, GoalView
from iris.harness import AgentRunner, SessionManager
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunOptions, FinishRun, RunCommit, RunLimits, RunPhase, RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.runtime import RuntimeEnvironment
from iris.store import SQLiteStore

from .fakes import BlockingProvider, FrozenClock, StaticProvider, text_response, tool_response
from .test_command_settlement import ControlledService, bind_service
from .test_goal_execution import goal_config


def _config(tmp_path: Path) -> AgentConfig:
    config = goal_config(tmp_path)
    return config.model_copy(
        update={"tools": config.tools.model_copy(update={"builtin": ["human.ask"]})}
    )


async def _wait_view(
    events: AsyncIterator[object], predicate: Callable[[GoalView], bool]
) -> GoalView:
    """从实际 mixed 通知等待指定状态，不靠轮询改变系统进展。"""
    async with asyncio.timeout(5):
        async for fact in events:
            if isinstance(fact, GoalChanged) and predicate(fact.view):
                return fact.view
    raise AssertionError("事件流结束前没有目标状态")


async def _save_waiting(
    tmp_path: Path,
    *,
    deadline: bool = False,
) -> tuple[Path, str, str, FrozenClock]:
    """留下已关闭进程资源、但仍有原 HITL 的 SQLite Goal Run。"""
    path = tmp_path / "goal.db"
    store = SQLiteStore(path)
    clock = FrozenClock()
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "继续吗？"},
            )
        )
    )
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store, clock=clock)
    options = AgentRunOptions(
        limits=RunLimits(
            deadline_at=clock.now() + timedelta(seconds=20) if deadline else None,
        )
    )
    goal = runner._goal_service.create("s", "回答后继续目标", max_rounds=1, run_options=options)
    result = await runner._start_goal_managed(goal.ref, run_id="original")
    assert result.run.phase is RunPhase.WAITING
    interaction_id = result.pending_interaction.interaction_id
    await runner.aclose()
    return path, goal.goal_id, interaction_id, clock


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_before_answer", [False, True])
async def test_sqlite_waiting_attach_keeps_run_and_original_answer(
    tmp_path: Path,
    pause_before_answer: bool,
) -> None:
    path, goal_id, interaction_id, clock = await _save_waiting(tmp_path)
    provider = StaticProvider(text_response("已处理回答"))
    store = SQLiteStore(path)
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store, clock=clock)
    manager = SessionManager(runner, "s")
    events = manager.events()
    try:
        before = await manager.goal.get()
        assert not before.armed and before.run.run_id == "original"
        assert not provider.requests
        attached = await manager.goal.resume()
        assert attached.disposition == "waiting"
        assert attached.view.goal.revision == before.goal.revision
        assert attached.view.interaction.interaction_id == interaction_id
        assert not provider.requests
        if pause_before_answer:
            await manager.goal.pause(reason="只处理这次回答")
        result = await manager.resume(
            interaction_id=interaction_id, response=QuestionInteractionResponse(answer="继续")
        )
        assert result.run.run_id == "original"
        assert result.run.stop_reason is RunStopReason.COMPLETED
        view = await _wait_view(
            events, lambda item: item.run is None and item.goal.status == "paused"
        )
        assert view.goal.goal_id == goal_id and view.goal.rounds_started == 1
        assert view.goal.reason.code == ("user" if pause_before_answer else "round_limit")
        assert not view.armed
        assert store.load_interaction(interaction_id).response.answer == "继续"
        assert len(provider.requests) == 1
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_restart_waiting_deadline_or_cleanup_failure_visible_without_publisher(
    tmp_path: Path,
    cleanup_fails: bool,
) -> None:
    path, goal_id, _, clock = await _save_waiting(tmp_path, deadline=True)
    store = SQLiteStore(path)
    provider = StaticProvider()
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store, clock=clock)
    service = ControlledService()
    service.fail = cleanup_fails
    if not cleanup_fails:
        service.release()
    bind_service(runner, service)
    manager = SessionManager(runner, "s")
    events = manager.events()
    try:
        assert not (await manager.goal.get()).armed
        attached = await manager.goal.resume()
        assert attached.disposition == "waiting"
        assert "original" in runner._command_lifecycle.deadlines
        clock.advance(seconds=21)
        if cleanup_fails:
            failed = await _wait_view(events, lambda item: item.driver_error is not None)
            assert not failed.armed and failed.run.phase is RunPhase.WAITING
            assert failed.goal.goal_id == goal_id and failed.goal.rounds_started == 1
            assert store.load_session_lane("s") == "original"
            assert "original" in runner._command_lifecycle.pending
            service.fail = False
            service.release()
            retried = await manager.goal.resume()
            assert retried.disposition == "stopped"
        stopped = await _wait_view(
            events, lambda item: item.run is None and item.goal.status == "paused"
        )
        assert not stopped.armed and stopped.goal.reason.code == "deadline_exceeded"
        assert stopped.goal.rounds_started == 1
        assert store.load_result("original").run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
        assert not provider.requests
    finally:
        service.fail = False
        service.release()
        await manager.close(cancel_run=True)
        await runner.aclose()


class _CompletingProvider(StaticProvider):
    """首个响应读实际 Goal revision 申报完成，第二个响应结束本 Run。"""

    def __init__(self, store: SQLiteStore) -> None:
        super().__init__()
        self.store = store

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            goal = self.store.get_current_goal("s")
            return tool_response(
                ToolUseBlock(
                    id="report",
                    name="report_goal",
                    input={
                        "goal_id": goal.goal_id,
                        "revision": goal.revision,
                        "decision": "complete",
                        "reason": "已完成实际要求",
                    },
                )
            )
        return text_response("完成")


async def _save_active(tmp_path: Path) -> tuple[Path, str, str]:
    """真实中断 provider 调用，保存尚未终态的原 Goal Run。"""
    path = tmp_path / "active.db"
    store = SQLiteStore(path)
    blocked = BlockingProvider()
    first = AgentRunner.from_config(_config(tmp_path), provider=blocked, store=store)
    goal = first._goal_service.create("s", "恢复原执行", max_rounds=1)
    task = asyncio.create_task(first._start_goal_managed(goal.ref, run_id="original"))
    await blocked.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await first.aclose()
    return path, goal.goal_id, store.load_run("original").current_activation_id


@pytest.mark.asyncio
async def test_active_restart_requires_fence_and_preserves_same_run_after_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, goal_id, _ = await _save_active(tmp_path)
    store = SQLiteStore(path)
    provider = _CompletingProvider(store)
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store)
    manager = SessionManager(runner, "s")
    events = manager.events()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def prepare(self: RuntimeEnvironment) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()

    try:
        missing = await manager.goal.resume()
        assert missing.disposition == "needs_recovery" and not missing.view.armed
        fence = missing.view.run.current_activation_id
        before = missing.view.goal
        with pytest.raises(IrisRunConflictError):
            await manager.goal.resume(expected_activation_id="stale")
        assert store.get_goal(goal_id) == before
        runner._prepared = False
        monkeypatch.setattr(RuntimeEnvironment, "aprepare", prepare)
        recovering = asyncio.create_task(manager.goal.resume(expected_activation_id=fence))
        await asyncio.wait_for(entered.wait(), 2)
        assert not recovering.done() and not provider.requests
        assert store.get_goal(goal_id) == before
        release.set()
        await asyncio.wait_for(recovering, 3)
        completed = await _wait_view(events, lambda item: item.goal.status == "completed")
        assert completed.goal.rounds_started == 1 and calls == 1
        assert store.get_goal_run("original").round_no == 1
        assert len(provider.requests) == 2
    finally:
        release.set()
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["pause", "close"])
async def test_goal_control_does_not_wait_for_explicit_recovery_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    """显式恢复的慢准备不占控制锁，期间停止后也不能在接手完成时重新 arm。"""
    path, _, fence = await _save_active(tmp_path)
    provider = BlockingProvider()
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=SQLiteStore(path))
    runner._prepared = False
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(self: RuntimeEnvironment) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(RuntimeEnvironment, "aprepare", prepare)
    manager = SessionManager(runner, "s")
    recovering = asyncio.create_task(manager.goal.resume(expected_activation_id=fence))
    stopping: asyncio.Task[object] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        stopping = asyncio.create_task(
            manager.goal.pause(reason="恢复准备期间暂停") if control == "pause" else manager.close()
        )
        await asyncio.wait_for(asyncio.shield(stopping), 0.5)
        assert not release.is_set() and not provider.requests
        assert runner._prepare_task is not None and not runner._prepare_task.cancelled()
        release.set()
        try:
            await asyncio.wait_for(asyncio.shield(recovering), 2)
        except asyncio.CancelledError:
            assert control == "close"
        assert runner._read_goal_process_state("s").armed_goal_id is None
        if control == "pause":
            assert (await manager.goal.get()).goal.status == "paused"
    finally:
        release.set()
        provider.release.set()
        await asyncio.gather(
            recovering, *(() if stopping is None else (stopping,)), return_exceptions=True
        )
        await asyncio.gather(*tuple(manager._managed_tasks), return_exceptions=True)
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_final_round_outcome_ready_keeps_current_report_without_model_call(
    tmp_path: Path,
) -> None:
    """上轮过时原因不能在纯恢复时改变版本，最后额度仍用已提交报告完成。"""

    class FailLastFinish(SQLiteStore):
        failed = False

        def finish_run(self, command: FinishRun) -> RunCommit:
            if command.run_id == "last" and not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("last finish interrupted")
            return super().finish_run(command)

    path = tmp_path / "finalize.db"
    store = FailLastFinish(path)

    class SupersededProvider(_CompletingProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            if len(self.requests) == 1:
                self.requests.append(request)
                return tool_response(ToolUseBlock(id="later-work", name="missing_work", input={}))
            return await super().complete(request)

    first = AgentRunner.from_config(
        _config(tmp_path), provider=SupersededProvider(store), store=store
    )
    goal = first._goal_service.create("s", "跨两轮完成", max_rounds=2)
    await first._start_goal_managed(goal.ref, run_id="earlier")
    current = first._goal_service.settle_run("earlier", now=first._now()).goal
    assert current.status == "active" and current.reason.code == "report_superseded"
    await first.aclose()
    second = AgentRunner.from_config(
        _config(tmp_path), provider=_CompletingProvider(store), store=store
    )
    with pytest.raises(IrisRunPersistenceError):
        await second._start_goal_managed(current.ref, run_id="last")
    pending = store.get_goal(goal.goal_id)
    assert pending.rounds_started == pending.max_rounds == 2
    assert pending.reason.code == "report_superseded"
    fence = store.load_run("last").current_activation_id
    await second.aclose()
    provider = StaticProvider()
    reopened = SQLiteStore(path)
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=reopened)
    manager = SessionManager(runner, "s")
    try:
        view = await manager.goal.get()
        assert not view.armed and view.goal == pending
        result = await manager.goal.resume(expected_activation_id=fence)
        assert result.disposition == "stopped"
        assert result.view.goal.status == "completed"
        assert result.view.goal.rounds_started == 2
        assert reopened.get_goal_run("last").applied_report_call_id == "report"
        assert not provider.requests
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()
