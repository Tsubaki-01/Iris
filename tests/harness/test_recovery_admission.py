"""Root 快照准入与实际后台作业的互斥。"""

import asyncio
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisCommandCleanupError, IrisRunStateError
from iris.harness import AgentRunner, SessionManager
from iris.harness.recovery_point import RootRecoveryCoordinator

from .fakes import StaticProvider


def recovery_runner(tmp_path: Path, *, enabled: bool = True, goal: bool = False) -> AgentRunner:
    """构造不访问外部服务的 root。"""
    return AgentRunner.from_config(
        AgentConfig.model_validate(
            {
                "name": "recovery",
                "model": "openai/test",
                "system": "test",
                "permissions": {"workspace": str(tmp_path)},
                "environment_recovery": {"enabled": enabled},
                "goal": {"enabled": goal},
            }
        ),
        provider=StaticProvider(),
    )


@pytest.mark.asyncio
async def test_optional_owner_and_normal_operations_overlap(tmp_path: Path) -> None:
    disabled = recovery_runner(tmp_path / "disabled", enabled=False)
    assert disabled._recovery is None
    runner = recovery_runner(tmp_path / "enabled")
    recovery = runner._recovery
    assert isinstance(recovery, RootRecoveryCoordinator)
    with recovery.operation(), recovery.operation():
        with pytest.raises(IrisRunStateError):
            recovery.begin_snapshot()
    recovery.begin_snapshot()
    with pytest.raises(IrisRunStateError):
        with recovery.operation():
            pass
    recovery.release_execution()
    await disabled.aclose()
    await runner.aclose()


@pytest.mark.asyncio
async def test_internal_waiters_resume_or_receive_cleanup_error(tmp_path: Path) -> None:
    runner = recovery_runner(tmp_path)
    recovery = runner._recovery
    recovery.begin_snapshot()
    entered = asyncio.Event()

    async def internal() -> None:
        async with recovery.internal_operation():
            entered.set()

    task = asyncio.create_task(internal())
    await asyncio.sleep(0)
    assert not entered.is_set()
    recovery.release_execution()
    await task
    recovery.begin_snapshot()
    blocked = asyncio.create_task(internal())
    await asyncio.sleep(0)
    recovery.block_execution(IrisCommandCleanupError("unpause failed"))
    with pytest.raises(IrisCommandCleanupError):
        await blocked
    await runner.aclose()


@pytest.mark.asyncio
async def test_actual_task_outlives_waiter(tmp_path: Path) -> None:
    runner = recovery_runner(tmp_path)
    gate = asyncio.Event()
    task = asyncio.create_task(gate.wait())
    runner._recovery.track_task(task)
    with pytest.raises(IrisRunStateError):
        runner._recovery.begin_snapshot()
    gate.set()
    await task
    runner._recovery.begin_snapshot()
    runner._recovery.release_execution()
    await runner.aclose()


@pytest.mark.asyncio
async def test_manager_rejects_before_mutation(tmp_path: Path) -> None:
    runner = recovery_runner(tmp_path)
    manager = SessionManager(runner, session_id="session")
    before = manager.snapshot()
    runner._recovery.begin_snapshot()
    with pytest.raises(IrisRunStateError):
        await manager.submit("test")
    assert manager.snapshot() == before
    runner._recovery.release_execution()
    await manager.close()
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["edit", "pause", "complete", "clear", "resume", "create"])
async def test_goal_mutations_fail_before_goal_state_changes(tmp_path: Path, action: str) -> None:
    runner = recovery_runner(tmp_path, goal=True)
    manager = SessionManager(runner, session_id="session")
    service = runner._goal_service
    goal = service.create_validated("session", service.parse_create("original", max_rounds=1))
    runner._recovery.begin_snapshot()
    actions = {
        "edit": lambda: manager.goal.edit(objective="changed"),
        "pause": lambda: manager.goal.pause(reason="paused"),
        "complete": lambda: manager.goal.complete(reason="done"),
        "clear": manager.goal.clear,
        "resume": manager.goal.resume,
        "create": lambda: manager.goal.create("replacement"),
    }
    with pytest.raises(IrisRunStateError):
        await actions[action]()
    assert service.get_current("session") == goal
    assert manager._goal_control.error is None
    runner._recovery.release_execution()
    await manager.close()
    await runner.aclose()


@pytest.mark.asyncio
async def test_prepare_waiter_cancellation_keeps_actual_work_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = recovery_runner(tmp_path)
    runner._prepared = False
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare() -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(runner, "_prepare_owned_environment", prepare)
    task = asyncio.create_task(runner.aprepare())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(IrisRunStateError):
        runner._recovery.begin_snapshot()
    release.set()
    await runner._prepare_task
    runner._recovery.begin_snapshot()
    runner._recovery.release_execution()
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("goal", [False, True])
async def test_deadline_and_follow_up_wait_without_losing_queue(tmp_path: Path, goal: bool) -> None:
    from datetime import UTC, datetime, timedelta

    from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
    from iris.message import ToolUseBlock

    from .fakes import text_response, tool_response
    from .test_goal_controls import _until

    config = AgentConfig.model_validate(
        {
            "name": "wait",
            "model": "openai/test",
            "system": "test",
            "permissions": {"workspace": str(tmp_path)},
            "environment_recovery": {"enabled": True},
            "tools": {"builtin": ["human.ask"]},
            "goal": {"enabled": goal},
        }
    )
    runner = AgentRunner.from_config(
        config,
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"})
            ),
            text_response("follow up"),
        ),
    )
    deadline = datetime.now(UTC) + timedelta(milliseconds=150)
    waiting = await runner.start(
        AgentRunRequest(input="ask", session_id="s"),
        options=AgentRunOptions(limits=RunLimits(deadline_at=deadline)),
    )
    manager = SessionManager(runner, "s")
    await manager.restore(waiting.run.run_id)
    queued = await manager.submit("after deadline", mode="follow_up")
    runner._recovery.begin_snapshot()
    await _until(lambda: runner._command_lifecycle.deadlines[waiting.run.run_id].fired)
    assert runner.get_run(waiting.run.run_id).phase is RunPhase.WAITING
    assert runner.store.load_run(queued.run_id) is None
    runner._recovery.release_execution()
    await _until(lambda: runner.store.load_result(queued.run_id) is not None)
    assert runner.get_result(waiting.run.run_id).run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    assert runner.store.load_run(waiting.run.run_id).options.limits.deadline_at == deadline
    assert runner.get_result(queued.run_id).run.stop_reason is RunStopReason.COMPLETED
    await manager.close()
    await runner.aclose()


@pytest.mark.asyncio
async def test_restore_resume_cancel_and_close_respect_snapshot_boundary(tmp_path: Path) -> None:
    from iris.hitl import QuestionInteractionResponse

    runner = recovery_runner(tmp_path)
    manager = SessionManager(runner, "s")
    before = manager.snapshot()
    runner._recovery.begin_snapshot()
    with pytest.raises(IrisRunStateError, match="快照"):
        await manager.restore("missing")
    with pytest.raises(IrisRunStateError, match="快照"):
        await manager.admit_resume(
            interaction_id="missing", response=QuestionInteractionResponse(answer="yes")
        )
    with pytest.raises(IrisRunStateError, match="快照"):
        runner.request_cancel("missing")
    assert manager.snapshot() == before
    closing = asyncio.create_task(manager.close(cancel_run=True))
    await asyncio.sleep(0)
    assert not manager._closed and not closing.done()
    runner._recovery.release_execution()
    await closing
    assert manager._close_complete
    await runner.aclose()


@pytest.mark.asyncio
async def test_cancelled_image_import_remains_admitted_until_thread_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import threading

    from PIL import Image

    from iris.harness import runner as runner_module

    from .test_goal_controls import _until

    runner = recovery_runner(tmp_path)
    data = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(data, format="PNG")
    entered, release = threading.Event(), threading.Event()
    save = runner_module.save_image

    def delayed(source: object, *, cache_dir: Path) -> object:
        entered.set()
        assert release.wait(5)
        return save(source, cache_dir=cache_dir)

    monkeypatch.setattr(runner_module, "save_image", delayed)
    task = asyncio.create_task(runner.import_image(data.getvalue(), session_id="s"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        with pytest.raises(IrisRunStateError):
            runner._recovery.begin_snapshot()
    finally:
        release.set()
    await _until(lambda: runner._recovery._operations == 0)
    runner._recovery.begin_snapshot()
    runner._recovery.release_execution()
    assert list((tmp_path / ".iris/image-cache").rglob("*.png"))
    await runner.aclose()
