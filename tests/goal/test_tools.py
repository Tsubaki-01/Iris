"""模型 Goal 工具只读投影与申报边界。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.goal import GoalProcessState, GoalService, GoalStatus
from iris.goal.store import AdmitGoalRun
from iris.goal.tools import GetGoalTool, ReportGoalTool
from iris.lifecycle import AgentRunOptions, RunPhase, RunStopReason
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolExecutionContext
from tests.store.test_lifecycle_store_contract import _create_command, _finish, _suspend


def _context(run_id: str = "run-1") -> ToolExecutionContext:
    return ToolExecutionContext(
        call_id="report-1",
        session_id="session-1",
        workspace_root=Path.cwd(),
        metadata={"run_id": run_id},
    )


def test_goal_tool_schemas_are_immediately_visible_and_keep_history() -> None:
    service = GoalService(InMemoryLifecycleStore())
    get, report = GetGoalTool(service), ReportGoalTool(service)
    assert get.name == "get_goal" and report.name == "report_goal"
    for tool in (get, report):
        assert tool.definition.deferred is False
        assert tool.definition.context_retention == "keep"
    assert get.input_schema["properties"] == {}
    assert set(report.input_schema["properties"]) == {"goal_id", "revision", "decision", "reason"}
    with pytest.raises(ValidationError):
        get.validate_input({"goal_id": "model cannot choose a session"})
    with pytest.raises(ValidationError):
        report.validate_input({"goal_id": "g", "revision": 1, "decision": "pause", "reason": "x"})


@pytest.mark.asyncio
async def test_get_goal_reports_absent_and_current_process_state_without_writing() -> None:
    store = InMemoryLifecycleStore()
    service = GoalService(store)
    tool = GetGoalTool(service)
    absent = await tool.arun(tool.validate_input({}), _context())
    assert absent.data["goal"] is None
    assert absent.data["armed"] is False
    goal = service.create("session-1", "完成任务")
    service.process_state_reader = lambda session_id: GoalProcessState(armed_goal_id=goal.goal_id)
    current = await tool.arun(tool.validate_input({}), _context("ordinary"))
    assert current.data["goal"]["goal_id"] == goal.goal_id
    assert current.data["armed"] is True
    assert store.get_goal(goal.goal_id) == goal


@pytest.mark.asyncio
async def test_report_returns_structured_claim_without_changing_goal() -> None:
    store = InMemoryLifecycleStore()
    service = GoalService(store)
    goal = service.create("session-1", "完成任务")
    admission = store.admit_goal_run(AdmitGoalRun(expected=goal.ref, create_run=_create_command()))
    tool = ReportGoalTool(service)
    args = {
        "goal_id": goal.goal_id,
        "revision": admission.goal.revision,
        "decision": "complete",
        "reason": "相关测试通过",
    }
    result = await tool.arun(tool.validate_input(args), _context())
    assert not result.is_error
    assert result.data["goal_report"] == args
    assert "正常结束后结算" in result.model_content
    assert store.get_goal(goal.goal_id) == admission.goal
    assert admission.goal.status is GoalStatus.ACTIVE


@pytest.mark.asyncio
async def test_report_rejects_ordinary_stale_and_replaced_goal() -> None:
    store = InMemoryLifecycleStore()
    service = GoalService(store)
    goal = service.create("session-1", "完成任务")
    tool = ReportGoalTool(service)
    args = {
        "goal_id": goal.goal_id,
        "revision": goal.revision,
        "decision": "complete",
        "reason": "done",
    }
    ordinary = await tool.arun(tool.validate_input(args), _context("ordinary"))
    assert ordinary.is_error and ordinary.error.code == "GOAL_STATE_ERROR"
    admission = store.admit_goal_run(AdmitGoalRun(expected=goal.ref, create_run=_create_command()))
    stale = await tool.arun(tool.validate_input(args), _context())
    assert stale.is_error and stale.error.code == "GOAL_CONFLICT"
    service.clear("session-1", expected=admission.goal.ref)
    replacement = service.create("session-1", "新目标")
    args.update(goal_id=replacement.goal_id, revision=replacement.revision)
    replaced = await tool.arun(tool.validate_input(args), _context())
    assert replaced.is_error and replaced.error.code == "GOAL_CONFLICT"
    assert store.get_goal(replacement.goal_id) == replacement


def test_get_view_shows_terminal_unsettled_run_without_reconciling() -> None:
    store = InMemoryLifecycleStore()
    service = GoalService(store)
    goal = service.create("session-1", "完成任务")
    admission = store.admit_goal_run(AdmitGoalRun(expected=goal.ref, create_run=_create_command()))
    assert service.get_view("session-1").settlement_pending is False
    _finish(store, admission.commit, RunStopReason.COMPLETED)
    view = service.get_view("session-1")
    assert view.settlement_pending is True
    assert view.run.run_id == "run-1"
    assert view.run_goal_id == goal.goal_id
    assert store.get_goal_run("run-1").settled_at is None
    assert store.get_goal(goal.goal_id) == admission.goal
    settlements = service.reconcile("session-1")
    assert len(settlements) == 1
    assert service.get_view("session-1").settlement_pending is False
    assert service.reconcile("session-1") == ()


def test_get_view_preserves_waiting_run_and_typed_interaction() -> None:
    store = InMemoryLifecycleStore()
    service = GoalService(store)
    goal = service.create("session-1", "完成任务")
    admission = store.admit_goal_run(AdmitGoalRun(expected=goal.ref, create_run=_create_command()))
    waiting = _suspend(store, admission.commit)
    view = service.get_view("session-1")
    assert view.run.phase is RunPhase.WAITING
    assert view.interaction == waiting.interaction
    assert view.run_goal_id == goal.goal_id
    assert view.settlement_pending is False


def test_run_options_validation_only_runs_at_create_or_explicit_option_edit() -> None:
    checked: list[AgentRunOptions] = []
    service = GoalService(InMemoryLifecycleStore(), run_options_validator=checked.append)
    goal = service.create("session-1", "完成任务")
    assert checked == [goal.run_options]
    edited = service.edit(goal.ref, objective="新的目标")
    assert len(checked) == 1
    service.edit(edited.ref, run_options=goal.run_options)
    assert len(checked) == 2
