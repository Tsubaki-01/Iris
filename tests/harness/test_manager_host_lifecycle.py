"""Host observation、原子文本路由与关闭的组合回归。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisRunStateError
from iris.harness import AgentRunner, AgentRunOptions, RunPhase, SessionManager
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import (
    BlockingProvider,
    RecordingPublisher,
    StaticProvider,
    build_runtime,
    tool_response,
)
from .test_session_manager import _wait_until


@pytest.mark.asyncio
async def test_broker_only_runs_and_submissions_do_not_require_mixed_consumer(
    tmp_path: Path,
) -> None:
    """已发布的 busy lifecycle 不占用未开启的本地 buffer。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    publisher = RecordingPublisher()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "broker-only",
        observation_mode="broker_only",
        submission_publisher=publisher,
        max_tracked_durable_runs=1,
        max_buffered_submission_events=2,
    )
    assert manager._event_buffer is None
    with pytest.raises(IrisRunStateError, match="broker_only"):
        manager.events()
    for index in range(3):
        current = await manager.submit(f"开始 {index}")
        await manager.submit("调整", mode="steer")
        provider.release.set()
        await _wait_until(lambda: manager._current_run_id is None)
        assert store.load_result(current.run_id).run.stop_reason is RunStopReason.COMPLETED
        provider.release.clear()
    assert [fact.event.state for fact in publisher.facts] == ["pending", "delivered"] * 3
    await manager.close()


@pytest.mark.asyncio
async def test_auto_submit_routes_from_current_durable_state(tmp_path: Path) -> None:
    """UI 无需等待 terminal event 就能提交下一轮或 steer。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store), "auto"
    )
    first = await manager.submit("开始", mode="auto", options=AgentRunOptions())
    steer = await manager.submit("调整", mode="auto", options=AgentRunOptions())
    assert steer.mode == "steer" and steer.run_id == first.run_id
    provider.release.set()
    await _wait_until(lambda: store.load_result(first.run_id) is not None)
    second = await manager.submit("下一轮", mode="auto", options=AgentRunOptions())
    assert second.mode is None and second.run_id != first.run_id
    await manager.close(cancel_run=True)


@pytest.mark.asyncio
async def test_close_with_cancel_does_not_launch_waiting_follow_up(tmp_path: Path) -> None:
    """先关闭 admission，再结算 waiting run，follow-up 从未开始。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    provider = StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry), store=store
    )
    manager = SessionManager(runner, "closing")
    current = await manager.submit("开始")
    await _wait_until(lambda: runner.get_run(current.run_id).phase is RunPhase.WAITING)
    follow_up = await manager.submit("下一轮", mode="follow_up")
    await manager.close(cancel_run=True)
    assert runner.get_run(current.run_id).stop_reason is RunStopReason.CANCELLED
    assert store.load_run(follow_up.run_id) is None
    assert store.load_session_lane("closing") is None
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_resume_admission_returns_before_provider_and_can_be_interrupted(
    tmp_path: Path,
) -> None:
    """远端 resume 确认 admission 后即返回，完整执行仍由 manager 持有。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    blocking = BlockingProvider()

    class WaitingThenBlocking:
        """先请求人工，再阻塞 resumed provider。"""

        first = True

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """恢复后的 provider 保持运行以验证即时回执。"""
            if self.first:
                self.first = False
                return tool_response(ToolUseBlock(id="write", name="write", input={}))
            return await blocking.complete(request)

    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=WaitingThenBlocking(), registry=registry),
        store=store,
    )
    manager = SessionManager(runner, "resume")
    current = await manager.submit("开始")
    await _wait_until(lambda: runner.get_run(current.run_id).phase is RunPhase.WAITING)
    interaction = runner.get_result(current.run_id).pending_interaction
    receipt = await asyncio.wait_for(
        manager.admit_resume(
            interaction_id=interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        ),
        timeout=1,
    )
    assert receipt.run_id == current.run_id
    assert receipt.interaction_id == interaction.interaction_id
    await asyncio.wait_for(blocking.started.wait(), timeout=1)
    await manager.close(cancel_run=True)
    assert runner.get_run(current.run_id).stop_reason is RunStopReason.CANCELLED
