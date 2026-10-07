"""SessionManager 控制投影的 admission 与收尾契约。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iris.harness import AgentRunner, SessionManager
from iris.harness.streaming import LiveFact, SessionControlChanged
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunPhase, RunResult
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response
from .test_session_manager import _wait_until


class RecordingPublisher:
    """保存原始只读事实。"""

    def __init__(self) -> None:
        self.facts: list[LiveFact] = []

    def publish(self, fact: LiveFact) -> None:
        """收集事实。"""
        self.facts.append(fact)


@pytest.mark.asyncio
async def test_snapshot_during_start_admission_and_pending_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读快照不等待 admission 锁，claim 未确认的正文仍可恢复。"""
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    publisher = RecordingPublisher()
    manager = SessionManager(runner, "control", submission_publisher=publisher)
    original = runner._start_managed
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(request: AgentRunRequest, **kwargs: object) -> RunResult:
        entered.set()
        await release.wait()
        return await original(request, **kwargs)

    monkeypatch.setattr(runner, "_start_managed", blocked)
    submit = asyncio.create_task(manager.submit("开始"))
    await entered.wait()
    snapshot = manager.snapshot()
    assert manager._lock.locked()
    assert snapshot.driver_state == "admitting"
    assert snapshot.run is None
    assert [(item.input, item.stage) for item in snapshot.pending] == [("开始", "admitting")]
    assert manager.snapshot() is snapshot
    release.set()
    receipt = await submit
    await provider.started.wait()
    queued = await manager.submit("新方向", mode="steer")
    follow_up = await manager.submit("下一轮", mode="follow_up")
    before_claim = manager.snapshot()
    run = runner.get_run_control(receipt.run_id)
    claimed = await manager._steering.claim(receipt.run_id, run.current_activation_id)
    assert claimed is not None
    assert [(item.submission_id, item.stage) for item in manager.snapshot().pending] == [
        (queued.submission_id, "committing"),
        (follow_up.submission_id, "queued"),
    ]
    assert before_claim.pending[0].stage == "queued"
    manager._steering.acknowledge(queued.submission_id)
    assert [item.submission_id for item in manager.snapshot().pending] == [follow_up.submission_id]
    assert "interrupt" in manager.snapshot().allowed_commands
    provider.release.set()
    await _wait_until(lambda: manager.snapshot().current_run_id is None)
    await manager.close()
    assert manager.snapshot().driver_state == "closed"
    assert not manager.snapshot().pending
    changes = [fact for fact in publisher.facts if isinstance(fact, SessionControlChanged)]
    assert changes[-1].snapshot is manager.snapshot()
    assert [fact.snapshot.revision for fact in changes] == sorted(
        {fact.snapshot.revision for fact in changes}
    )


@pytest.mark.asyncio
async def test_waiting_ready_follows_task_settlement_and_resume_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WAITING 已提交而旧 task 未退出时不能 resume，恢复准入中快照仍可读。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "ok", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write", name="write", input={})),
                text_response("完成"),
            ),
        ),
        store=InMemoryLifecycleStore(),
    )
    original = runner._start_managed
    settling, release = asyncio.Event(), asyncio.Event()

    async def delayed(request: AgentRunRequest, **kwargs: object) -> RunResult:
        result = await original(request, **kwargs)
        settling.set()
        await release.wait()
        return result

    monkeypatch.setattr(runner, "_start_managed", delayed)
    manager = SessionManager(runner, "waiting")
    receipt = await manager.submit("写入")
    await settling.wait()
    assert manager.snapshot().run.phase is RunPhase.WAITING
    assert manager.snapshot().driver_state == "settling"
    assert "resume" not in manager.snapshot().allowed_commands
    release.set()
    await _wait_until(lambda: "resume" in manager.snapshot().allowed_commands)
    original_resume = runner._resume_managed
    entered, resume_release = asyncio.Event(), asyncio.Event()

    async def delayed_resume(run_id: str, **kwargs: object) -> RunResult:
        entered.set()
        await resume_release.wait()
        return await original_resume(run_id, **kwargs)

    monkeypatch.setattr(runner, "_resume_managed", delayed_resume)
    result = runner.get_result(receipt.run_id)
    resume = asyncio.create_task(
        manager.admit_resume(
            interaction_id=result.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    )
    await entered.wait()
    assert manager.snapshot().driver_state == "admitting"
    assert "resume" not in manager.snapshot().allowed_commands
    resume_release.set()
    await resume
    await _wait_until(lambda: manager.snapshot().current_run_id is None)
    await manager.close()


@pytest.mark.asyncio
async def test_follow_up_admission_retains_pending_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow-up 已出 FIFO、尚未完成 create 时仍有且仅有一个 pending。"""
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    manager = SessionManager(runner, "follow-up-control")
    await manager.submit("开始")
    await provider.started.wait()
    follow_up = await manager.submit("下一轮", mode="follow_up")
    original = runner._start_managed
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(request: AgentRunRequest, **kwargs: object) -> RunResult:
        entered.set()
        await release.wait()
        return await original(request, **kwargs)

    monkeypatch.setattr(runner, "_start_managed", delayed)
    provider.release.set()
    await entered.wait()
    snapshot = manager.snapshot()
    assert snapshot.current_run_id == follow_up.run_id
    assert snapshot.driver_state == "admitting"
    assert [(item.input, item.stage) for item in snapshot.pending] == [("下一轮", "admitting")]
    release.set()
    await _wait_until(lambda: manager.snapshot().current_run_id is None)
    await manager.close()
    assert manager.snapshot().driver_state == "closed"
    assert not manager.snapshot().pending


@pytest.mark.asyncio
async def test_cancelled_submit_waiter_does_not_leave_delivered_input_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """取消 HTTP 等待者后，manager 持有的任务继续并在准入完成时清除 pending。"""
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    manager = SessionManager(runner, "cancelled-waiter")
    original = runner._start_managed
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(request: AgentRunRequest, **kwargs: object) -> RunResult:
        entered.set()
        await release.wait()
        return await original(request, **kwargs)

    monkeypatch.setattr(runner, "_start_managed", blocked)
    waiter = asyncio.create_task(manager.submit("已接纳输入"))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await provider.started.wait()
    assert manager.snapshot().driver_state == "running"
    assert manager.snapshot().pending == ()
    assert runner.get_session("cancelled-waiter").messages[0].text == "已接纳输入"
    provider.release.set()
    await _wait_until(lambda: manager.snapshot().current_run_id is None)
    await manager.close()


@pytest.mark.asyncio
async def test_control_snapshot_does_not_read_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读取只返回最近投影，不查库或 reconcile。"""
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    manager = SessionManager(runner, "snapshot")

    def unexpected(run_id: str) -> None:
        raise AssertionError("snapshot must not read the store")

    monkeypatch.setattr(runner, "get_run", unexpected)
    assert manager.snapshot().driver_state == "idle"
    assert "submit" in manager.snapshot().allowed_commands
    await manager.close()
