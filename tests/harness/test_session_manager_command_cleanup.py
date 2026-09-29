"""Manager 的失败清理重试与不同 run 的取消 owner。"""

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisCommandCleanupError, IrisRunStateError
from iris.harness import AgentRunner, SessionManager
from iris.lifecycle import RunPhase, RunResult, RunStopReason
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, tool_response
from .test_command_settlement import ControlledService, FailedProvider, bind_service
from .test_session_manager import _wait_until


@pytest.mark.asyncio
async def test_active_cleanup_failure_interrupt_retries_before_follow_up(tmp_path: Path) -> None:
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailedProvider()), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    service.fail = True
    service.fail_release = asyncio.Event()
    bind_service(runner, service)
    manager = SessionManager(runner, "session")
    current = await manager.submit("first")
    task = manager._current_task
    assert task is not None
    await service.entered.wait()
    follow_up = await manager.submit("next", mode="follow_up")
    service.fail_release.set()
    with pytest.raises(IrisCommandCleanupError):
        await task
    await _wait_until(lambda: manager._current_task is None)
    assert runner.get_run(current.run_id).phase is RunPhase.ACTIVE
    service.fail = False
    service.release()
    try:
        await manager.interrupt()
        await _wait_until(lambda: runner.get_run(current.run_id).phase is RunPhase.TERMINAL)
        assert runner.get_result(current.run_id).error.code == "PROVIDER_ERROR"
        await _wait_until(lambda: runner.store.load_run(follow_up.run_id) is not None)
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


@pytest.mark.asyncio
async def test_manager_close_cleanup_failure_can_retry_without_reopening_admission(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    service = ControlledService()
    service.fail = True
    bind_service(runner, service)
    manager = SessionManager(runner, "session")

    async def consume() -> None:
        async for _ in manager.events():
            pass

    consumer = asyncio.create_task(consume())
    current = await manager.submit("first")
    await provider.started.wait()
    with pytest.raises(IrisCommandCleanupError):
        await manager.close(cancel_run=True)
    await asyncio.wait_for(consumer, 1)
    assert runner.get_run(current.run_id).phase is RunPhase.ACTIVE
    with pytest.raises(IrisRunStateError, match="closed"):
        await manager.submit("no admission")
    service.fail = False
    service.release()
    try:
        await manager.close(cancel_run=True)
        assert runner.get_run(current.run_id).stop_reason is RunStopReason.CANCELLED
        calls = len(service.calls)
        await manager.close(cancel_run=True)
        assert len(service.calls) == calls
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_old_interrupt_delivery_does_not_own_next_run_and_close_waits_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "write", name="write", capabilities={ToolCapability.WRITE})
    response = tool_response(ToolUseBlock(id="write", name="write", input={}))
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path, registry=registry, provider=StaticProvider(response, response)
        ),
        store=InMemoryLifecycleStore(),
    )
    manager = SessionManager(runner, "session")
    first = await manager.submit("first")
    await manager._current_task
    follow_up = await manager.submit("next", mode="follow_up")
    first_terminal = asyncio.Event()
    release_first = asyncio.Event()
    second_cancel = asyncio.Event()
    original = runner.cancel

    async def cancel(run_id: str, *, reason: str | None = None) -> RunResult:
        result = await original(run_id, reason=reason)
        if run_id == first.run_id:
            first_terminal.set()
            await release_first.wait()
        else:
            second_cancel.set()
        return result

    monkeypatch.setattr(runner, "cancel", cancel)
    close_task: asyncio.Task[None] | None = None
    try:
        await manager.interrupt()
        await first_terminal.wait()
        # 旧 cancel 尚未完成事件投递，但 durable terminal 已允许下一条 follow-up。
        await manager.submit("steer next", mode="steer")
        await _wait_until(
            lambda: (
                (run := runner.store.load_run(follow_up.run_id)) is not None
                and run.phase is RunPhase.WAITING
            )
        )
        await manager.interrupt()
        await asyncio.wait_for(second_cancel.wait(), 1)
        close_task = asyncio.create_task(manager.close(cancel_run=True))
        await asyncio.sleep(0)
        assert not close_task.done()
        release_first.set()
        await asyncio.wait_for(close_task, 1)
    finally:
        release_first.set()
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await manager.close(cancel_run=True)
        await runner.aclose()
