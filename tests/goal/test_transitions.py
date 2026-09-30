"""Goal 纯状态转换的核心规则。"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from iris.exceptions import IrisGoalConflictError, IrisGoalStateError
from iris.goal import GoalReason, GoalRef, GoalSnapshot, GoalStatus
from iris.goal.store import (
    AdmitGoalRun,
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    EditGoal,
    PauseGoal,
    ResumeGoal,
)
from iris.goal.transitions import (
    admit_goal_snapshot,
    apply_goal_update,
    create_goal_snapshot,
    require_current_goal,
    require_goal_creation,
)
from iris.lifecycle import AgentRunOptions, AgentRunRequest, CreateRun, RunCheckpoint

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = _NOW + timedelta(seconds=1)
_REASON = GoalReason(code="user", text="用户操作")


def _goal() -> GoalSnapshot:
    return create_goal_snapshot(
        CreateGoal(
            goal_id="g",
            session_id="s",
            objective="完成任务",
            max_rounds=2,
            run_options=AgentRunOptions(),
            now=_NOW,
        )
    )


def _ref(goal: GoalSnapshot) -> GoalRef:
    return GoalRef(goal_id=goal.goal_id, revision=goal.revision)


def test_create_and_replace_current_goal_rules() -> None:
    goal = _goal()
    assert goal.status is GoalStatus.ACTIVE
    assert goal.revision == goal.rounds_started == 0
    require_goal_creation(None)
    with pytest.raises(IrisGoalStateError):
        require_goal_creation(goal)
    completed = apply_goal_update(
        goal, CompleteGoal(expected=_ref(goal), reason=_REASON, now=_LATER)
    )
    require_goal_creation(completed)


def test_current_goal_cas_checks_identity_revision_and_selection() -> None:
    goal = _goal()
    require_current_goal(goal, _ref(goal), is_current=True)
    for expected, current in [
        (GoalRef(goal_id="other", revision=0), True),
        (GoalRef(goal_id="g", revision=1), True),
        (_ref(goal), False),
    ]:
        with pytest.raises(IrisGoalConflictError):
            require_current_goal(goal, expected, is_current=current)


def test_pause_resume_edit_preserve_consumed_rounds() -> None:
    goal = _goal().model_copy(update={"rounds_started": 1})
    paused = apply_goal_update(goal, PauseGoal(expected=_ref(goal), reason=_REASON, now=_LATER))
    assert paused.status is GoalStatus.PAUSED
    assert paused.revision == 1
    assert (
        apply_goal_update(paused, PauseGoal(expected=_ref(paused), reason=_REASON, now=_LATER))
        is paused
    )
    resumed = apply_goal_update(paused, ResumeGoal(expected=_ref(paused), now=_LATER))
    assert resumed.status is GoalStatus.ACTIVE
    assert resumed.revision == 2
    assert resumed.reason is None
    assert apply_goal_update(resumed, ResumeGoal(expected=_ref(resumed), now=_LATER)) is resumed
    edited = apply_goal_update(
        resumed, EditGoal(expected=_ref(resumed), objective="新目标", now=_LATER)
    )
    assert edited.objective == "新目标"
    assert edited.status is GoalStatus.PAUSED
    assert edited.rounds_started == 1
    assert edited.revision == 3
    assert edited.created_at == goal.created_at


def test_completed_cannot_be_resumed_paused_or_edited() -> None:
    goal = apply_goal_update(
        _goal(), CompleteGoal(expected=_ref(_goal()), reason=_REASON, now=_LATER)
    )
    for command in [
        PauseGoal(expected=_ref(goal), reason=_REASON, now=_LATER),
        ResumeGoal(expected=_ref(goal), now=_LATER),
        EditGoal(expected=_ref(goal), objective="other", now=_LATER),
    ]:
        with pytest.raises(IrisGoalStateError):
            apply_goal_update(goal, command)
    assert (
        apply_goal_update(goal, CompleteGoal(expected=_ref(goal), reason=_REASON, now=_LATER))
        is goal
    )
    cleared = apply_goal_update(goal, ClearGoal(session_id="s", expected=_ref(goal), now=_LATER))
    assert cleared.status is GoalStatus.COMPLETED
    assert cleared.revision == goal.revision + 1


def test_change_of_pause_reason_is_a_real_mutation() -> None:
    command = PauseGoal(expected=_ref(_goal()), reason=_REASON, now=_LATER)
    paused = apply_goal_update(_goal(), command)
    changed = apply_goal_update(
        paused, replace(command, reason=GoalReason(code="user", text="新原因"))
    )
    assert changed.revision == paused.revision + 1


def test_exhausted_goal_only_resumes_existing_run() -> None:
    goal = _goal().model_copy(update={"rounds_started": 2})
    paused = apply_goal_update(goal, ResumeGoal(expected=_ref(goal), now=_LATER))
    assert paused.status is GoalStatus.PAUSED
    assert paused.reason is not None and paused.reason.code == "round_limit"
    assert paused.rounds_started == 2
    resumed = apply_goal_update(
        paused,
        ResumeGoal(expected=_ref(paused), now=_LATER),
        has_resumable_run=True,
    )
    assert resumed.status is GoalStatus.ACTIVE
    with pytest.raises(IrisGoalStateError):
        apply_goal_update(resumed, EditGoal(expected=_ref(resumed), max_rounds=1, now=_LATER))
    edited = apply_goal_update(resumed, EditGoal(expected=_ref(resumed), max_rounds=2, now=_LATER))
    assert edited.max_rounds == edited.rounds_started == 2


@pytest.mark.parametrize(("rounds_started", "has_resumable_run"), [(1, False), (2, True)])
def test_active_resume_preserves_report_reason_and_revision(
    rounds_started: int,
    has_resumable_run: bool,
) -> None:
    """纯 arm/恢复不能清除诊断并让同轮已提交的报告版本失效。"""
    goal = _goal().model_copy(
        update={
            "revision": 4,
            "rounds_started": rounds_started,
            "reason": GoalReason(code="report_superseded", text="上轮报告已过时"),
        }
    )
    resumed = apply_goal_update(
        goal,
        ResumeGoal(expected=goal.ref, now=_LATER),
        has_resumable_run=has_resumable_run,
    )
    assert resumed is goal


def test_admission_consumes_round_and_preserves_input_revision_in_binding() -> None:
    goal = _goal()
    create = CreateRun(
        request=AgentRunRequest(input="继续", session_id="s", run_id="r"),
        options=goal.run_options,
        agent_id="agent",
        start_activation_id="a",
        now=_LATER,
        initial_checkpoint=RunCheckpoint(
            run_id="r",
            sequence=1,
            activation_id="a",
            engine_cursor={},
            session_revision=0,
            model_steps_reserved=0,
            model_steps_committed=0,
        ),
    )
    command = AdmitGoalRun(expected=_ref(goal), create_run=create)
    admitted, binding = admit_goal_snapshot(goal, command)
    assert admitted.rounds_started == binding.round_no == 1
    assert admitted.revision == 1
    assert binding.admission_revision == goal.revision == 0
    assert binding.run_id == "r"
    assert admitted.updated_at == _LATER
    for changed in [
        goal.model_copy(update={"status": GoalStatus.PAUSED}),
        goal.model_copy(update={"rounds_started": 2}),
    ]:
        with pytest.raises(IrisGoalStateError):
            admit_goal_snapshot(changed, command)
