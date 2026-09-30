"""Goal 管理所需的恢复 hooks 与后台事实通知，不依赖 live publisher。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.exceptions import IrisGoalStateError, IrisRunPersistenceError
from iris.harness import AgentRunner, CommandCleanupFailed
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    FinishRun,
    RunCommit,
    RunEvent,
    RunEventKind,
    RunLimits,
    RunPhase,
    RunStopReason,
)
from iris.message import Msg, ToolUseBlock
from iris.runtime import RuntimeEnvironment, SteeringInput
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import BlockingProvider, StaticProvider, build_runtime, text_response, tool_response
from .test_command_settlement import ControlledService, bind_service


class _Steering:
    """恢复仍使用原管理调用的 steering 端口。"""

    def __init__(self) -> None:
        self.pending: SteeringInput | None = SteeringInput(
            submission_id="steer",
            message=Msg.user("恢复后补充要求"),
        )
        self.delivered: list[str] = []

    async def claim(self, run_id: str, activation_id: str) -> SteeringInput | None:
        """只交付一次输入。"""
        item, self.pending = self.pending, None
        return item

    def acknowledge(self, submission_id: str) -> None:
        """记录真实 committed acknowledgement。"""
        self.delivered.append(submission_id)

    def fail(self, submission_id: str, reason: str) -> None:
        """这些场景不应拒绝输入。"""
        raise AssertionError(f"{submission_id}: {reason}")


def test_late_attachment_observes_existing_collector_and_exact_unregister(tmp_path: Path) -> None:
    """旧 collector 在发生事实时查 root attachment，close 不误删新 owner。"""
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    collector = runner._event_collector()
    facts: list[RunEvent | CommandCleanupFailed] = []
    callback = facts.append
    runner._register_session_fact_callback("s", callback)
    with pytest.raises(IrisGoalStateError):
        runner._register_session_fact_callback("s", callback)
    runner._unregister_session_fact_callback("s", lambda fact: None)
    event = RunEvent(
        run_id="r",
        session_id="s",
        sequence=1,
        kind=RunEventKind.RUN_TERMINAL,
        occurred_at=runner._now(),
    )
    collector.record((event, event))
    assert facts == [event]
    runner._unregister_session_fact_callback("s", callback)
    collector.record((event.model_copy(update={"sequence": 2}),))
    assert facts == [event]


@pytest.mark.asyncio
async def test_managed_recover_preserves_hooks_after_shared_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare 后重新读取旧 fence，继续同一 managed callback、signal 与 steering。"""
    store = InMemoryLifecycleStore()
    blocked = BlockingProvider()
    first = AgentRunner(runtime=build_runtime(tmp_path, provider=blocked), store=store)
    abandoned = asyncio.create_task(first.start(AgentRunRequest(input="开始", run_id="recover")))
    await blocked.started.wait()
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    fence = store.load_run("recover").current_activation_id
    provider = StaticProvider(text_response("恢复"), text_response("已处理补充要求"))
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    runner._prepared = False
    preparing, release = asyncio.Event(), asyncio.Event()
    prepare_calls = 0

    async def prepare(self: RuntimeEnvironment) -> None:
        nonlocal prepare_calls
        prepare_calls += 1
        preparing.set()
        await release.wait()

    monkeypatch.setattr(RuntimeEnvironment, "aprepare", prepare)
    started = asyncio.Event()
    events: list[RunEvent] = []
    steering = _Steering()
    task = asyncio.create_task(
        runner._recover_managed(
            "recover",
            expected_activation_id=fence,
            steering=steering,
            durable_event_callback=events.append,
            activation_started=started,
        )
    )
    await asyncio.wait_for(preparing.wait(), 1)
    assert not started.is_set()
    release.set()
    result = await asyncio.wait_for(task, 2)
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert prepare_calls == 1
    assert started.is_set()
    assert steering.delivered == ["steer"]
    assert any(event.kind is RunEventKind.ACTIVATION_ABANDONED for event in events)
    assert events[-1].kind is RunEventKind.RUN_TERMINAL
    assert len({event.sequence for event in events}) == len(events)
    await runner.aclose()


@pytest.mark.asyncio
async def test_managed_finalize_relays_terminal_without_starting_engine(tmp_path: Path) -> None:
    """已生成结果的恢复直接形成终态，仍能唤醒 managed 与 root 通知。"""

    class FailFinishOnce(InMemoryLifecycleStore):
        failed = False

        def finish_run(self, command: FinishRun) -> RunCommit:
            if not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("finish interrupted")
            return super().finish_run(command)

    store = FailFinishOnce()
    first = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    with pytest.raises(IrisRunPersistenceError):
        await first.start(AgentRunRequest(input="完成", run_id="finalize"))
    provider = StaticProvider()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    facts: list[RunEvent | CommandCleanupFailed] = []
    runner._register_session_fact_callback("default", facts.append)
    managed: list[RunEvent] = []
    result = await runner._recover_managed(
        "finalize",
        expected_activation_id=store.load_run("finalize").current_activation_id,
        durable_event_callback=managed.append,
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert not provider.requests
    assert managed[-1].kind is RunEventKind.RUN_TERMINAL
    assert facts == managed
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_waiting_deadline_notifies_without_live_publisher(
    tmp_path: Path,
    cleanup_fails: bool,
) -> None:
    """后台期限终态或排空失败都通知当前 attachment，失败保留 lane 等待重试。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    provider = StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry),
        store=InMemoryLifecycleStore(),
    )
    service = ControlledService()
    service.fail = cleanup_fails
    if not cleanup_fails:
        service.release()
    bind_service(runner, service)
    facts: list[RunEvent | CommandCleanupFailed] = []
    observed = asyncio.Event()

    def callback(fact: RunEvent | CommandCleanupFailed) -> None:
        facts.append(fact)
        if isinstance(fact, CommandCleanupFailed) or fact.kind is RunEventKind.RUN_TERMINAL:
            observed.set()

    runner._register_session_fact_callback("default", callback)
    waiting = await runner.start(
        AgentRunRequest(input="写入", run_id="waiting"),
        options=AgentRunOptions(
            limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(milliseconds=100))
        ),
    )
    assert waiting.run.phase is RunPhase.WAITING
    await asyncio.wait_for(observed.wait(), 2)
    if cleanup_fails:
        assert isinstance(facts[-1], CommandCleanupFailed)
        assert runner.get_run("waiting").phase is RunPhase.WAITING
        assert runner.store.load_session_lane("default") == "waiting"
        assert "waiting" in runner._command_lifecycle.pending
        service.fail = False
        service.release()
        await runner.recover("waiting")
    assert runner.get_result("waiting").run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    terminal = [
        fact
        for fact in facts
        if isinstance(fact, RunEvent) and fact.kind is RunEventKind.RUN_TERMINAL
    ]
    assert len(terminal) == 1
    await runner.aclose()
