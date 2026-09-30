"""Goal 单轮执行在输入恢复、失败和有效工具配置处的边界。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import pytest

from iris.context import ContextSection, ContextSlot
from iris.exceptions import IrisConfigError, IrisGoalStateError, IrisRunStateError
from iris.goal.models import GoalAdmission
from iris.goal.store import AdmitGoalRun
from iris.harness import AgentRunner
from iris.harness.runner import ActiveActivation
from iris.lifecycle import (
    AgentRunOptions,
    RunLimits,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
)
from iris.memory import MemoryOverviewDocument, MemoryService, SQLiteMemoryStore
from iris.store import InMemoryLifecycleStore

from .fakes import FrozenClock, StaticProvider, text_response
from .test_goal_execution import goal_config


class _PublishedOverview(MemoryService):
    """提供确定性已发布概览，记录新窗口实际读取次数。"""

    def __init__(self, path: Path) -> None:
        super().__init__(SQLiteMemoryStore(path))
        self.reads = 0

    async def aload_overviews(
        self, namespaces: Sequence[str]
    ) -> tuple[MemoryOverviewDocument, ...]:
        self.reads += 1
        return (
            MemoryOverviewDocument(
                namespace="project",
                path=Path("Memory.md"),
                source_revision=7,
                text="稳定记忆正文",
                navigation="项目测试资料",
            ),
        )


@pytest.mark.asyncio
async def test_before_input_recovery_preserves_bci_memory_window_and_single_goal_input(
    tmp_path: Path,
) -> None:
    config = goal_config(tmp_path)
    config = config.model_copy(
        update={"memory": config.memory.model_copy(update={"enabled": True})}
    )
    store = InMemoryLifecycleStore()
    memory = _PublishedOverview(tmp_path / "memory.db")
    first = AgentRunner.from_config(
        config, provider=StaticProvider(), store=store, memory_service=memory
    )
    goal = first._goal_service.create("s", "恢复目标执行")
    admitted, cursor = first._admit_goal_start(goal.ref, run_id="goal-before-input")
    assert cursor.position == "before_input"
    assert store.load_session("s").messages == []
    assert store.load_session("s").context_window is None
    await first.aclose()

    provider = StaticProvider(text_response())
    second = AgentRunner.from_config(config, provider=provider, store=store, memory_service=memory)
    second.runtime.environment.context_input = second.runtime.environment.context_input.model_copy(
        update={
            "before_current_input": ContextSection(
                slots=[ContextSlot(name="bci", content="当前环境资料")],
            )
        }
    )
    result = await second.recover(
        admitted.commit.run.run_id,
        expected_activation_id=admitted.commit.run.current_activation_id,
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    session = store.load_session("s")
    bci = [
        message
        for message in session.messages
        if message.metadata.get("context_kind") == "before_current_input"
    ]
    inputs = [
        message
        for message in session.messages
        if message.metadata.get("context_kind") == "goal_continuation"
    ]
    assert len(bci) == len(inputs) == 1
    assert "当前环境资料" in bci[0].text
    assert inputs[0].sender == "context"
    assert inputs[0].metadata["goal_id"] == goal.goal_id
    assert inputs[0].metadata["round_no"] == 1
    assert session.context_window is not None
    assert "稳定记忆正文" in session.context_window.memory_overview
    assert session.context_window.sources[0].source_revision == 7
    assert memory.reads == 1
    assert "稳定记忆正文" in provider.requests[0].messages[0].text
    assert any("当前环境资料" in message.text for message in provider.requests[0].messages)
    assert store.load_run_context(
        result.run.run_id, include_tool_discovery=False
    ).protected_indices == (0, 1)
    assert store._sessions["s"].read_state.last_ordinary_user_index is None
    assert store.get_goal(goal.goal_id).rounds_started == 1
    assert await second.recover(result.run.run_id) == result
    assert store.load_session("s") == session
    assert memory.reads == len(provider.requests) == 1
    await second.aclose()


@pytest.mark.asyncio
async def test_before_input_recovery_rejects_new_tool_choice_without_changing_run(
    tmp_path: Path,
) -> None:
    store = InMemoryLifecycleStore()
    config = goal_config(tmp_path)
    first = AgentRunner.from_config(config, provider=StaticProvider(), store=store)
    goal = first._goal_service.create("s", "恢复原目标")
    admitted, _ = first._admit_goal_start(goal.ref, run_id="needs-valid-tools")
    await first.aclose()
    forced = config.model_copy(
        update={"model": config.model.model_copy(update={"tool_choice": "required"})}
    )
    provider = StaticProvider()
    second = AgentRunner.from_config(forced, provider=provider, store=store)
    with pytest.raises(IrisConfigError):
        await second.recover(
            admitted.commit.run.run_id,
            expected_activation_id=admitted.commit.run.current_activation_id,
        )
    assert store.load_run(admitted.commit.run.run_id) == admitted.commit.run
    assert store.get_goal(goal.goal_id) == admitted.goal
    assert store.get_goal_run(admitted.commit.run.run_id) == admitted.binding
    assert provider.requests == []
    await second.aclose()


@pytest.mark.asyncio
async def test_empty_goal_edit_does_not_pause_or_advance_revision(tmp_path: Path) -> None:
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=StaticProvider())
    service = runner._goal_service
    goal = service.create("s", "保留原目标")
    with pytest.raises(IrisGoalStateError, match="至少提供"):
        service.edit(goal.ref)
    assert service.get(goal.goal_id) == goal
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["prepare", "admission"])
async def test_failure_before_admission_does_not_consume_goal_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    provider = StaticProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    goal = runner._goal_service.create("s", "完成目标")

    async def fail_prepare() -> None:
        raise IrisGoalStateError("injected before admission")

    def fail_admission(command: AdmitGoalRun) -> GoalAdmission:
        raise IrisGoalStateError("injected before admission")

    if boundary == "prepare":
        monkeypatch.setattr(runner, "aprepare", fail_prepare)
    else:
        monkeypatch.setattr(store, "admit_goal_run", fail_admission)
    with pytest.raises(IrisGoalStateError, match="injected before admission"):
        await runner._start_goal_managed(goal.ref, run_id="not-created")
    assert store.load_run("not-created") is None
    assert store.get_goal_run("not-created") is None
    assert store.load_session_lane("s") is None
    assert store.get_goal(goal.goal_id) == goal
    assert provider.requests == []
    await runner.aclose()


@pytest.mark.asyncio
async def test_registration_failure_keeps_admitted_run_for_exact_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StaticProvider(text_response())
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(goal_config(tmp_path), provider=provider, store=store)
    goal = runner._goal_service.create("s", "完成目标")
    started = asyncio.Event()

    def fail_register(active: ActiveActivation, expected_activation_id: str | None) -> None:
        raise IrisRunStateError("injected live registration failure")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_register", fail_register)
        with pytest.raises(IrisRunStateError, match="injected live registration failure"):
            await runner._start_goal_managed(
                goal.ref, run_id="admitted", activation_started=started
            )
    original = store.load_run("admitted")
    assert original is not None and original.phase is RunPhase.ACTIVE
    assert store.load_session_lane("s") == original.run_id
    assert store.get_goal_run(original.run_id).round_no == 1
    assert store.get_goal(goal.goal_id).rounds_started == 1
    assert not started.is_set() and provider.requests == []
    result = await runner.recover(
        original.run_id, expected_activation_id=original.current_activation_id
    )
    assert result.run.run_id == original.run_id
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert store.get_goal(goal.goal_id).rounds_started == 1
    assert len(provider.requests) == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_expired_absolute_deadline_consumes_admission_without_provider_call(
    tmp_path: Path,
) -> None:
    clock = FrozenClock()
    provider = StaticProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(
        goal_config(tmp_path), provider=provider, store=store, clock=clock
    )
    deadline = clock.now() - timedelta(seconds=1)
    goal = runner._goal_service.create(
        "s",
        "已经到期",
        run_options=AgentRunOptions(limits=RunLimits(deadline_at=deadline)),
    )
    result = await runner._start_goal_managed(goal.ref, run_id="expired-goal")
    assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
    assert result.run.limits.deadline_at == deadline
    assert store.get_goal(goal.goal_id).rounds_started == 1
    assert store.get_goal_run(result.run.run_id).round_no == 1
    assert provider.requests == []
    assert (
        runner._goal_service.settle_run(result.run.run_id, now=clock.now()).goal.status == "paused"
    )
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["auto", None])
async def test_runtime_tool_choice_override_controls_create_edit_and_execution(
    tmp_path: Path,
    override: str | None,
) -> None:
    config = goal_config(tmp_path)
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"tool_choice": "required"})}
    )
    provider = StaticProvider(text_response())
    runner = AgentRunner.from_config(config, provider=provider)
    service = runner._goal_service
    with pytest.raises(IrisConfigError):
        service.create("s", "配置要求强制工具时不能直接运行 Goal")
    assert service.get_current("s") is None
    options = AgentRunOptions(
        runtime=RuntimeExecutionOptions(request_options={"tool_choice": override})
    )
    goal = service.create("s", "运行覆盖允许自主申报", run_options=options)
    for invalid in (
        {},
        {"tool_choice": "none"},
        {"tool_choice": {"type": "function", "function": {"name": "get_goal"}}},
    ):
        with pytest.raises(IrisConfigError):
            service.edit(
                goal.ref,
                run_options=AgentRunOptions(
                    runtime=RuntimeExecutionOptions(request_options=invalid)
                ),
            )
        assert service.get(goal.goal_id) == goal
    edited = service.edit(goal.ref, run_options=options)
    resumed = service.resume(edited.ref)
    result = await runner._start_goal_managed(resumed.ref, run_id="configured-goal")
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert provider.requests[0].tool_choice == override
    assert {tool["function"]["name"] for tool in provider.requests[0].tools} >= {
        "get_goal",
        "report_goal",
    }
    await runner.aclose()
