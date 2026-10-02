"""结束 Hook 与 Manager 输入等待、FIFO 和资源关闭的真实组合契约。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from iris.exceptions import IrisCommandCleanupError, IrisRunStateError
from iris.harness import AgentRunner, SessionManager
from iris.hooks import HookEvent, HookRegistration, RunFinishedEvent
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import DataBlock, ImageBlock, ImageFileRef, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response
from .test_command_settlement import ControlledService, bind_service
from .test_goal_execution import goal_config
from .test_session_manager import _wait_until


class FinishedGate:
    """只阻塞第一次真实结束通知，取消后的清理可独立放行。"""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.drained = asyncio.Event()
        self.drained.set()
        self.runs: list[str] = []
        self.failure: Exception | None = None

    async def __call__(self, event: HookEvent) -> None:
        """以外部事件证明调用、取消与真正收口。"""
        assert isinstance(event, RunFinishedEvent)
        self.runs.append(event.result.run.run_id)
        if len(self.runs) != 1:
            return
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.drained.wait()
            raise
        if self.failure is not None:
            raise self.failure


def _attach(runner: AgentRunner, gate: FinishedGate) -> None:
    """P5 使用内部环境接线；公共 YAML 配置留到 P6。"""
    dispatcher = HookDispatcher(
        [HookRegistration(event="run.finished", name="finished", handler=gate)]
    )
    runner.runtime = AgentRuntime(replace(runner.runtime.environment, hook_dispatcher=dispatcher))


def _manager(tmp_path: Path) -> tuple[SessionManager, StaticProvider, FinishedGate]:
    provider = StaticProvider(*(text_response(str(index)) for index in range(4)))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    gate = FinishedGate()
    _attach(runner, gate)
    return SessionManager(runner, "s"), provider, gate


async def _close(manager: SessionManager, gate: FinishedGate) -> None:
    """失败断言也放行测试持有的 handler，再按宿主关闭顺序回收。"""
    gate.release.set()
    gate.drained.set()
    await manager.close(cancel_run=True)
    await _wait_until(lambda: not manager._managed_tasks)
    await manager._runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "auto"])
@pytest.mark.parametrize("with_image", [False, True])
async def test_submit_waits_outside_lock_and_reuses_original_options(
    tmp_path: Path, mode: str | None, with_image: bool
) -> None:
    manager, provider, gate = _manager(tmp_path)
    next_submit: asyncio.Task | None = None
    try:
        first = await manager.submit("first")
        await asyncio.wait_for(gate.entered.wait(), 1)
        assert manager._runner.get_run(first.run_id).phase is RunPhase.TERMINAL
        options = AgentRunOptions(limits=RunLimits(max_model_steps=3))
        ref = ImageFileRef(path=tmp_path / "image.png", mime_type="image/png", width=1, height=1)
        content: str | list[DataBlock] = (
            [TextBlock(text="second"), ImageBlock(original=ref, model=ref)]
            if with_image
            else "second"
        )
        next_submit = asyncio.create_task(manager.submit(content, mode=mode, options=options))
        await asyncio.sleep(0)
        assert not next_submit.done() and len(provider.requests) == 1
        async with asyncio.timeout(1):
            async with manager._lock:
                assert manager._current_run_id == first.run_id
        with pytest.raises(IrisRunStateError):
            await manager.submit("cannot steer terminal", mode="steer")
        gate.release.set()
        second = await asyncio.wait_for(next_submit, 2)
        assert second.mode is None and second.run_id != first.run_id
        assert manager._runner.get_run(second.run_id).limits.max_model_steps == 3
        assert manager._runner.store.load_run(second.run_id).request.input == content
    finally:
        if next_submit is not None and not next_submit.done():
            next_submit.cancel()
            await asyncio.gather(next_submit, return_exceptions=True)
        await _close(manager, gate)


@pytest.mark.asyncio
@pytest.mark.parametrize("detach", [False, True])
async def test_waiter_cancellation_or_detach_does_not_cancel_finished_owner(
    tmp_path: Path, detach: bool
) -> None:
    manager, provider, gate = _manager(tmp_path)
    waiting: asyncio.Task | None = None
    try:
        await manager.submit("first")
        await asyncio.wait_for(gate.entered.wait(), 1)
        waiting = asyncio.create_task(manager.submit("second"))
        await asyncio.sleep(0)
        assert not waiting.done()
        if detach:
            await manager.close()
            with pytest.raises(IrisRunStateError):
                await asyncio.wait_for(waiting, 1)
        else:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        assert not gate.cancelled.is_set() and len(provider.requests) == 1
    finally:
        if waiting is not None and not waiting.done():
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        await _close(manager, gate)


@pytest.mark.asyncio
async def test_follow_up_fifo_waits_for_finished_and_is_actively_woken(tmp_path: Path) -> None:
    manager, provider, gate = _manager(tmp_path)
    try:
        first = await manager.submit("first")
        await asyncio.wait_for(gate.entered.wait(), 1)
        second = await manager.submit("second", mode="follow_up")
        third = await manager.submit("third", mode="follow_up")
        snapshot = await manager.interrupt()
        assert snapshot.run_id == first.run_id and snapshot.phase is RunPhase.TERMINAL
        assert manager._pending.follow_up_count == 2
        assert manager._runner.store.load_run(second.run_id) is None
        assert manager._runner.store.load_run(third.run_id) is None
        assert not gate.cancelled.is_set()
        gate.release.set()
        await _wait_until(lambda: len(gate.runs) == 3)
        assert gate.runs == [first.run_id, second.run_id, third.run_id]
        assert len(provider.requests) == 3
    finally:
        await _close(manager, gate)


@pytest.mark.asyncio
async def test_cancel_close_cancels_actual_finished_owner_and_waits_for_drain(
    tmp_path: Path,
) -> None:
    manager, provider, gate = _manager(tmp_path)
    closing: asyncio.Task | None = None
    try:
        first = await manager.submit("first")
        await asyncio.wait_for(gate.entered.wait(), 1)
        queued = await manager.submit("queued", mode="follow_up")
        gate.drained.clear()
        closing = asyncio.create_task(manager.close(cancel_run=True))
        await asyncio.wait_for(gate.cancelled.wait(), 1)
        assert not closing.done()
        assert manager._runner.get_result(first.run_id).run.stop_reason is RunStopReason.COMPLETED
        gate.drained.set()
        await asyncio.wait_for(closing, 2)
        assert manager._runner.store.load_run(queued.run_id) is None
        assert len(provider.requests) == 1
    finally:
        gate.drained.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await _close(manager, gate)


@pytest.mark.asyncio
async def test_cancel_close_finds_finished_without_a_manager_task(tmp_path: Path) -> None:
    """共享 owner 中的直接 SDK 结束任务也可被关闭找到，不依赖 current_task。"""
    manager, provider, gate = _manager(tmp_path)
    direct = asyncio.create_task(
        manager._runner.start(AgentRunRequest(input="direct", session_id="s"))
    )
    try:
        await asyncio.wait_for(gate.entered.wait(), 1)
        assert manager._current_run_id is None and not manager._managed_tasks
        await asyncio.wait_for(manager.close(cancel_run=True), 2)
        result = await asyncio.wait_for(direct, 1)
        assert gate.cancelled.is_set()
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert len(provider.requests) == 1
    finally:
        gate.release.set()
        await asyncio.gather(direct, return_exceptions=True)
        await _close(manager, gate)


@pytest.mark.asyncio
async def test_goal_waits_for_shared_finished_without_a_manager_owned_run(tmp_path: Path) -> None:
    """直接 SDK 的结束通知也阻止 Goal 首轮，不能只依赖 managed_tasks。"""
    provider = StaticProvider(text_response("ordinary"), text_response("goal"))
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    gate = FinishedGate()
    _attach(runner, gate)
    manager = SessionManager(runner, "s")
    direct = asyncio.create_task(runner.start(AgentRunRequest(input="ordinary", session_id="s")))
    try:
        await asyncio.wait_for(gate.entered.wait(), 1)
        assert not manager._managed_tasks and manager._current_run_id is None
        await manager.goal.create("one goal round", max_rounds=1)
        await asyncio.sleep(0)
        assert runner._goal_service.get_current("s").rounds_started == 0
        assert len(provider.requests) == 1
        gate.release.set()
        await asyncio.wait_for(direct, 2)
        await _wait_until(lambda: len(provider.requests) == 2)
        assert runner._goal_service.get_current("s").rounds_started == 1
    finally:
        gate.release.set()
        await asyncio.gather(direct, return_exceptions=True)
        await _close(manager, gate)


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_cancel_close_also_cancels_finished_started_during_close(
    tmp_path: Path, waiting: bool
) -> None:
    """从 ACTIVE/WAITING 关闭时，随后新登记的结束通知也归本次关闭收口。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "write", name="write", description="write", capabilities={ToolCapability.WRITE}
    )
    provider = (
        StaticProvider(tool_response(ToolUseBlock(id="w", name="write", input={})))
        if waiting
        else BlockingProvider()
    )
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=InMemoryLifecycleStore(),
    )
    gate = FinishedGate()
    _attach(runner, gate)
    manager = SessionManager(runner, "s")
    try:
        receipt = await manager.submit("first")
        if waiting:
            await _wait_until(lambda: runner.get_run(receipt.run_id).phase is RunPhase.WAITING)
            await _wait_until(lambda: not manager._managed_tasks)
        else:
            await asyncio.wait_for(provider.started.wait(), 1)
        await asyncio.wait_for(manager.close(cancel_run=True), 2)
        assert runner.get_result(receipt.run_id).run.stop_reason is RunStopReason.CANCELLED
        assert not runner._hook_lifecycle.has_pending("s")
    finally:
        await _close(manager, gate)


@pytest.mark.asyncio
async def test_finished_cleanup_failure_rejects_waiter_and_fails_queued_follow_up(
    tmp_path: Path,
) -> None:
    manager, provider, gate = _manager(tmp_path)
    service = ControlledService()
    service.fail = True
    service.release()
    bind_service(manager._runner, service)
    gate.failure = IrisCommandCleanupError("hook cleanup failed")
    waiting: asyncio.Task | None = None
    try:
        await manager.submit("first")
        await asyncio.wait_for(gate.entered.wait(), 1)
        queued = await manager.submit("queued", mode="follow_up")
        waiting = asyncio.create_task(manager.submit("waiter"))
        await asyncio.sleep(0)
        gate.release.set()
        with pytest.raises(IrisCommandCleanupError) as caught:
            await asyncio.wait_for(waiting, 2)
        assert caught.value is manager._runner._hook_lifecycle.admission_error
        await _wait_until(lambda: manager._pending.follow_up_count == 0)
        assert manager._runner.store.load_run(queued.run_id) is None
        assert len(provider.requests) == 1
    finally:
        service.fail = False
        if waiting is not None:
            await asyncio.gather(waiting, return_exceptions=True)
        await _close(manager, gate)


@pytest.mark.asyncio
async def test_goal_does_not_admit_next_round_after_finished_cleanup_failure(
    tmp_path: Path,
) -> None:
    provider = StaticProvider(text_response("first"), text_response("must not run"))
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider)
    gate = FinishedGate()
    gate.failure = IrisCommandCleanupError("hook cleanup failed")
    _attach(runner, gate)
    service = ControlledService()
    service.fail = True
    service.release()
    bind_service(runner, service)
    manager = SessionManager(runner, "s")
    try:
        await manager.goal.create("two rounds", max_rounds=2)
        await asyncio.wait_for(gate.entered.wait(), 1)
        assert runner._goal_service.get_current("s").rounds_started == 1
        gate.release.set()
        await _wait_until(lambda: runner._hook_lifecycle.admission_error is not None)
        await _wait_until(lambda: not manager._managed_tasks)
        view = await manager.goal.get()
        assert view.driver_error.code == "COMMAND_CLEANUP_FAILED" and not view.armed
        assert view.goal.rounds_started == 1 and len(provider.requests) == 1
        with pytest.raises(IrisCommandCleanupError) as caught:
            await manager.submit("new run")
        assert caught.value is runner._hook_lifecycle.admission_error
    finally:
        service.fail = False
        await _close(manager, gate)
