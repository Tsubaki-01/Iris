"""显式 restore 保持 Runner 的恢复裁决与 Manager 的唯一任务所有权。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisRunConflictError, IrisRunStateError
from iris.harness import AgentRunner, AgentRunRequest, RunPhase, SessionManager
from iris.hitl import PermissionInteractionResponse
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response
from .test_session_manager import _wait_until


@pytest.mark.asyncio
async def test_restore_pending_waiting_attaches_without_execution(tmp_path: Path) -> None:
    """读取和附着不调用模型；回答及排队输入沿同一 Manager 恢复。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "ok", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write", name="write", input={})),
        text_response("完成"),
        text_response("补充完成"),
    )
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry),
        store=InMemoryLifecycleStore(),
    )
    waiting = await runner.start(AgentRunRequest(input="写入", session_id="waiting"))
    manager = SessionManager(runner, "waiting")
    count = len(provider.requests)
    receipt = await manager.restore(waiting.run.run_id)
    assert receipt.disposition == "attached_waiting"
    assert "resume" in receipt.control.allowed_commands
    assert len(provider.requests) == count
    await manager.submit("补充", mode="steer")
    await manager.admit_resume(
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    await _wait_until(lambda: manager.snapshot().current_run_id is None)
    terminal = await manager.restore(waiting.run.run_id)
    assert terminal.disposition == "settled"
    assert terminal.control.current_run_id is None
    assert "补充" in [message.text for message in runner.get_session("waiting").messages]
    await manager.close()


@pytest.mark.asyncio
async def test_restore_same_run_with_done_task_really_recovers(tmp_path: Path) -> None:
    """残留 current_run_id 不产生伪幂等，只有存活 managed task 才复用。"""
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    manager = SessionManager(runner, "recover")
    initial = await manager.submit("开始")
    await provider.started.wait()
    task = manager._current_task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_until(lambda: manager.snapshot().driver_state == "detached")
    fence = runner.get_run(initial.run_id).current_activation_id
    with pytest.raises(IrisRunConflictError):
        await manager.restore(initial.run_id, expected_activation_id="wrong")
    receipt = await manager.restore(initial.run_id, expected_activation_id=fence)
    assert receipt.disposition == "recovery_started"
    assert receipt.control.run.current_activation_id != fence
    assert (await manager.restore(initial.run_id)).disposition == "already_managed"
    await manager.submit("恢复后补充", mode="steer")
    provider.release.set()
    await _wait_until(lambda: manager.snapshot().current_run_id is None)
    assert runner.get_run(initial.run_id).phase is RunPhase.TERMINAL
    await manager.close()


@pytest.mark.asyncio
async def test_restore_rejects_other_session(tmp_path: Path) -> None:
    """Run 身份固定于 bound session，不因 ID 已知而跨会话接管。"""
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    result = await runner.start(AgentRunRequest(input="完成", session_id="other"))
    manager = SessionManager(runner, "bound")
    with pytest.raises(IrisRunStateError):
        await manager.restore(result.run.run_id)
    assert manager.snapshot().current_run_id is None
    await manager.close()
