"""Goal 复用真实 memory 前台计数与闲置整理的交接契约。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisRunPersistenceError
from iris.goal.models import GoalAdmission, GoalRef
from iris.harness import AgentRunner, SessionManager
from iris.harness.runner import ActiveActivation
from iris.lifecycle import RunCommit, RunEventKind, RunPhase
from iris.memory import MemoryObserveInput, MemoryService, SQLiteMemoryStore
from iris.memory.generation_models import MemoryGenerationConfig
from iris.memory.mirror import FileMemoryMirror
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.runtime import RuntimeCursor
from iris.store import InMemoryLifecycleStore

from .fakes import StaticProvider, text_response, tool_response
from .test_goal_execution import goal_config
from .test_goal_session import goal_state


class _Main(StaticProvider):
    """首轮和后继各在真实 provider 入口等待测试放行。"""

    def __init__(self, store: InMemoryLifecycleStore) -> None:
        super().__init__()
        self.store = store
        self.entered = (asyncio.Event(), asyncio.Event())
        self.release = (asyncio.Event(), asyncio.Event())

    async def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(request)
        if index < 2:
            self.entered[index].set()
            await self.release[index].wait()
        goal = self.store.get_current_goal("s")
        if index == 1 and goal.rounds_started == 2:
            return tool_response(
                ToolUseBlock(
                    id="complete-goal",
                    name="report_goal",
                    input={
                        "goal_id": goal.goal_id,
                        "revision": goal.revision,
                        "decision": "complete",
                        "reason": "两轮完成",
                    },
                )
            )
        return text_response()


class _Generation(StaticProvider):
    """复用 memory capture 测试中的空提炼响应，记录真实后台入口。"""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        self.started.set()
        if request.model == "overview":
            return text_response('{"core_facts":"","knowledge_scope":"暂无记忆"}')
        return text_response('{"observations": []}')


def _runner(
    tmp_path: Path,
    *,
    idle_seconds: float = 0,
) -> tuple[AgentRunner, _Main, _Generation]:
    """沿用 memory capture 的真实服务装配，额外开启 Goal。"""
    store = InMemoryLifecycleStore()
    main, generation = _Main(store), _Generation()
    settings = MemoryGenerationConfig(enabled=True, idle_seconds=idle_seconds)
    memory = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror", workspace_root=tmp_path),
        generation_provider=generation,
        generation_model="generation",
        generation_config=settings,
        overview_provider=generation,
        overview_model="overview",
    )
    config = goal_config(tmp_path)
    config = config.model_copy(
        update={
            "memory": config.memory.model_copy(update={"enabled": True, "generation": settings}),
        }
    )
    runner = AgentRunner.from_config(config, provider=main, store=store, memory_service=memory)
    memory.observe(MemoryObserveInput(text="已有待整理材料"))
    return runner, main, generation


def _hold_first_cleanup(
    runner: AgentRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[asyncio.Event, asyncio.Event]:
    """让终态已提交但旧前台调用仍然存在，暴露临时预留窗口。"""
    entered, release = asyncio.Event(), asyncio.Event()
    original = runner._settle_live_resources

    async def cleanup(active: ActiveActivation) -> None:
        if not entered.is_set():
            entered.set()
            await release.wait()
        await original(active)

    monkeypatch.setattr(runner, "_settle_live_resources", cleanup)
    return entered, release


async def _close(
    manager: SessionManager, runner: AgentRunner, main: _Main, *, cancel_run: bool = True
) -> None:
    """测试退出时放行 provider 并按原宿主顺序关闭。"""
    for event in main.release:
        event.set()
    await asyncio.wait_for(manager.close(cancel_run=cancel_run), 5)
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_seconds", [0, 0.01])
async def test_continuous_goal_handoff_is_single_and_waits_for_activation_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_seconds: float,
) -> None:
    runner, main, generation = _runner(tmp_path, idle_seconds=idle_seconds)
    cleanup, release_cleanup = _hold_first_cleanup(runner, monkeypatch)
    registered, release_registration = asyncio.Event(), asyncio.Event()
    original = runner._run_created_start

    async def start(created: RunCommit, **kwargs: object) -> object:
        binding = runner.store.get_goal_run(created.run.run_id)
        if binding.round_no == 2:
            registered.set()
            await release_registration.wait()
        return await original(created, **kwargs)

    monkeypatch.setattr(runner, "_run_created_start", start)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("两轮完成", max_rounds=2)
        await asyncio.wait_for(main.entered[0].wait(), 5)
        main.release[0].set()
        await asyncio.wait_for(cleanup.wait(), 5)
        candidate = manager._goal_control.driver.intent
        assert candidate is not None
        assert manager._memory_handoffs == {candidate.run_id}
        before = runner._memory_maintenance._foreground
        terminal = next(
            event
            for event in runner.store.list_events(candidate.source_run_id)
            if event.kind is RunEventKind.RUN_TERMINAL
        )
        manager._relay_run_event(terminal)
        manager._relay_run_event(terminal)
        assert manager._goal_control.driver.intent is candidate
        assert runner._memory_maintenance._foreground == before == 2
        await asyncio.sleep(0.03)
        assert runner.store.load_run(candidate.run_id) is None
        release_cleanup.set()
        await asyncio.wait_for(registered.wait(), 5)
        assert runner.store.get_goal_run(candidate.run_id).round_no == 2
        assert manager._goal_control.driver.intent is None
        assert manager._memory_handoffs == {candidate.run_id}
        assert runner._memory_maintenance._foreground == 2
        await asyncio.sleep(0.03)
        assert generation.requests == []
        release_registration.set()
        await asyncio.wait_for(main.entered[1].wait(), 5)
        await asyncio.sleep(0.03)
        assert manager._memory_handoffs == set()
        assert runner._memory_maintenance._foreground == 1
        assert generation.requests == []
        main.release[1].set()
        completed = await goal_state(manager.events(), "completed")
        assert completed.goal.rounds_started == 2
        await asyncio.wait_for(generation.started.wait(), 5)
        assert manager._memory_handoffs == set()
        assert not runner._memory_maintenance.foreground_active
    finally:
        release_cleanup.set()
        release_registration.set()
        await _close(manager, runner, main)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["pause", "complete", "close"])
async def test_stopping_goal_candidate_releases_only_temporary_foreground(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    runner, main, generation = _runner(tmp_path)
    cleanup, release_cleanup = _hold_first_cleanup(runner, monkeypatch)
    manager = SessionManager(runner, "s")
    closing: asyncio.Task[None] | None = None
    try:
        created = await manager.goal.create("可撤销后继", max_rounds=3)
        await asyncio.wait_for(main.entered[0].wait(), 5)
        main.release[0].set()
        await asyncio.wait_for(cleanup.wait(), 5)
        candidate = manager._goal_control.driver.intent
        assert candidate is not None and manager._memory_handoffs == {candidate.run_id}
        if control == "close":
            closing = asyncio.create_task(manager.close(cancel_run=True))
            await asyncio.sleep(0)
        elif control == "pause":
            await manager.goal.pause(reason="用户暂停")
        else:
            await manager.goal.complete(reason="用户验收")
        assert manager._memory_handoffs == set()
        if control != "close":
            assert runner._memory_maintenance._foreground == 1
        assert runner.store.load_run(candidate.run_id) is None
        release_cleanup.set()
        if closing is not None:
            await asyncio.wait_for(closing, 5)
        await asyncio.wait_for(generation.started.wait(), 5)
        assert runner.store.get_goal(created.view.goal.goal_id).rounds_started == 1
        assert len(main.requests) == 1
        assert not runner._memory_maintenance.foreground_active
    finally:
        release_cleanup.set()
        await _close(manager, runner, main)


@pytest.mark.asyncio
async def test_goal_pause_does_not_release_user_follow_up_handoff(
    tmp_path: Path,
) -> None:
    runner, main, generation = _runner(tmp_path)
    manager = SessionManager(runner, "s")
    try:
        created = await manager.goal.create("用户仍可排队", max_rounds=3)
        await asyncio.wait_for(main.entered[0].wait(), 5)
        follow_up = await manager.submit("优先处理用户后继", mode="follow_up")
        manager._reserve_memory_handoff(follow_up.run_id)
        assert manager._memory_handoffs == {follow_up.run_id}
        assert manager._goal_control.driver.intent is None
        await manager.goal.pause(reason="暂停目标，保留用户输入")
        assert manager._memory_handoffs == {follow_up.run_id}
        assert runner._memory_maintenance._foreground == 2
        main.release[0].set()
        await asyncio.wait_for(main.entered[1].wait(), 5)
        assert runner.store.get_goal_run(follow_up.run_id) is None
        await asyncio.sleep(0.03)
        assert generation.requests == []
        main.release[1].set()
        await asyncio.wait_for(generation.started.wait(), 5)
        assert runner.store.get_goal(created.view.goal.goal_id).rounds_started == 1
        assert manager._memory_handoffs == set()
    finally:
        await _close(manager, runner, main)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_admission", [False, True])
async def test_goal_successor_failure_releases_handoff_without_refunding_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_admission: bool,
) -> None:
    runner, main, generation = _runner(tmp_path)
    original_admit = runner._admit_goal_start
    original_register = runner._register

    def admit(expected: GoalRef, *, run_id: str) -> tuple[GoalAdmission, RuntimeCursor]:
        if not after_admission and runner.store.get_goal(expected.goal_id).rounds_started:
            raise IrisRunPersistenceError("goal successor admission failed")
        return original_admit(expected, run_id=run_id)

    def register(active: ActiveActivation, expected_activation_id: str | None) -> None:
        if after_admission and runner.store.get_goal_run(active.run_id).round_no == 2:
            raise IrisRunPersistenceError("goal successor registration failed")
        original_register(active, expected_activation_id)

    monkeypatch.setattr(runner, "_admit_goal_start", admit)
    monkeypatch.setattr(runner, "_register", register)
    manager = SessionManager(runner, "s")
    try:
        created = await manager.goal.create("后继失败", max_rounds=3)
        await asyncio.wait_for(main.entered[0].wait(), 5)
        main.release[0].set()
        await asyncio.wait_for(generation.started.wait(), 5)
        assert manager._memory_handoffs == set()
        assert not runner._memory_maintenance.foreground_active
        view = await manager.goal.get()
        assert view.driver_error is not None and not view.armed
        expected_rounds = 2 if after_admission else 1
        assert runner.store.get_goal(created.view.goal.goal_id).rounds_started == expected_rounds
        assert len(runner.store._runs) == expected_rounds
        if after_admission:
            assert view.run.phase is RunPhase.ACTIVE
            assert runner.store.get_goal_run(view.run.run_id).round_no == 2
        else:
            assert view.goal.status == "paused" and view.goal.reason.code == "start_failed"
        assert len(main.requests) == 1
    finally:
        await _close(manager, runner, main, cancel_run=not after_admission)


@pytest.mark.asyncio
async def test_tracker_backpressure_releases_handoff_and_allows_idle_memory(tmp_path: Path) -> None:
    runner, main, generation = _runner(tmp_path)
    manager = SessionManager(runner, "s", max_tracked_durable_runs=1)
    completed: asyncio.Task[object] | None = None
    try:
        await manager.goal.create("释放tracker后继续", max_rounds=2)
        await asyncio.wait_for(main.entered[0].wait(), 5)
        main.release[0].set()
        await asyncio.wait_for(generation.started.wait(), 5)
        assert manager._memory_handoffs == set()
        assert not runner._memory_maintenance.foreground_active
        assert (await manager.goal.get()).goal.rounds_started == 1
        assert len(main.requests) == 1
        completed = asyncio.create_task(goal_state(manager.events(), "completed"))
        await asyncio.wait_for(main.entered[1].wait(), 5)
        main.release[1].set()
        view = await asyncio.wait_for(completed, 5)
        assert view.goal.rounds_started == 2
    finally:
        if completed is not None and not completed.done():
            completed.cancel()
            await asyncio.gather(completed, return_exceptions=True)
        await _close(manager, runner, main)
