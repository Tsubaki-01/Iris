"""Goal 状态转换；存储在持有事务或锁时应用一次领域检查。"""

from __future__ import annotations

from typing import cast

from ..exceptions import IrisGoalConflictError, IrisGoalStateError
from .models import GoalReason, GoalRef, GoalRunBinding, GoalSnapshot, GoalStatus
from .store import (
    AdmitGoalRun,
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    EditGoal,
    GoalUpdate,
    PauseGoal,
)

_ROUND_LIMIT_REASON = GoalReason(code="round_limit", text="目标自动执行轮数已用尽")
_EDIT_REASON = GoalReason(code="edited", text="目标已修改，等待恢复")


def require_goal_creation(current: GoalSnapshot | None) -> None:
    """创建不能隐式覆盖未完成的当前目标。"""
    if current is not None and current.status is not GoalStatus.COMPLETED:
        raise IrisGoalStateError("当前目标尚未完成，请先 clear", goal_id=current.goal_id)


def require_current_goal(goal: GoalSnapshot, expected: GoalRef, *, is_current: bool) -> None:
    """由存储在操作边界一次检查 current、身份与 Goal revision。"""
    if not is_current or goal.goal_id != expected.goal_id or goal.revision != expected.revision:
        raise IrisGoalConflictError("目标身份、版本或当前选择已变化", goal_id=expected.goal_id)


def create_goal_snapshot(command: CreateGoal) -> GoalSnapshot:
    """将已解析创建输入投影为初始快照。"""
    return GoalSnapshot.model_construct(
        goal_id=command.goal_id,
        session_id=command.session_id,
        revision=0,
        objective=command.objective,
        status=GoalStatus.ACTIVE,
        reason=None,
        max_rounds=command.max_rounds,
        rounds_started=0,
        run_options=command.run_options,
        created_at=command.now,
        updated_at=command.now,
    )


def apply_goal_update(
    goal: GoalSnapshot,
    command: GoalUpdate,
    *,
    has_resumable_run: bool = False,
) -> GoalSnapshot:
    """只验证当前状态与受影响字段，返回一次修订或原快照。"""
    if isinstance(command, ClearGoal):
        return goal.model_copy(update={"revision": goal.revision + 1, "updated_at": command.now})
    if isinstance(command, CompleteGoal):
        status, reason = GoalStatus.COMPLETED, command.reason
    else:
        if goal.status is GoalStatus.COMPLETED:
            raise IrisGoalStateError("已完成目标不能继续修改或推进", goal_id=goal.goal_id)
        if isinstance(command, EditGoal):
            if command.max_rounds is not None and command.max_rounds < goal.rounds_started:
                raise IrisGoalStateError("新额度不能小于已使用轮数", goal_id=goal.goal_id)
            return goal.model_copy(
                update={
                    "objective": goal.objective if command.objective is None else command.objective,
                    "max_rounds": goal.max_rounds
                    if command.max_rounds is None
                    else command.max_rounds,
                    "run_options": goal.run_options
                    if command.run_options is None
                    else command.run_options,
                    "status": GoalStatus.PAUSED,
                    "reason": _EDIT_REASON,
                    "revision": goal.revision + 1,
                    "updated_at": command.now,
                }
            )
        if isinstance(command, PauseGoal):
            status, reason = GoalStatus.PAUSED, command.reason
        elif not has_resumable_run and goal.rounds_started >= goal.max_rounds:
            status, reason = GoalStatus.PAUSED, _ROUND_LIMIT_REASON
        elif goal.status is GoalStatus.ACTIVE:
            return goal
        else:
            status, reason = GoalStatus.ACTIVE, None
    if goal.status is status and goal.reason == reason:
        return goal
    return goal.model_copy(
        update={
            "status": status,
            "reason": reason,
            "revision": goal.revision + 1,
            "updated_at": command.now,
        }
    )


def admit_goal_snapshot(
    goal: GoalSnapshot,
    command: AdmitGoalRun,
) -> tuple[GoalSnapshot, GoalRunBinding]:
    """构造一次准入的领域事实，lane/CAS/未结算检查由存储持锁完成。"""
    if goal.status is not GoalStatus.ACTIVE:
        raise IrisGoalStateError("只有 active 目标可以准入新轮", goal_id=goal.goal_id)
    if goal.rounds_started >= goal.max_rounds:
        raise IrisGoalStateError("目标自动执行轮数已用尽", goal_id=goal.goal_id)
    if command.create_run.request.session_id != goal.session_id:
        raise IrisGoalStateError("目标 Run 必须属于同一 session", goal_id=goal.goal_id)
    round_no = goal.rounds_started + 1
    updated = goal.model_copy(
        update={
            "rounds_started": round_no,
            "revision": goal.revision + 1,
            "updated_at": command.create_run.now,
        }
    )
    binding = GoalRunBinding.model_construct(
        run_id=cast(str, command.create_run.request.run_id),
        goal_id=goal.goal_id,
        round_no=round_no,
        admission_revision=command.expected.revision,
        settled_at=None,
        applied_report_call_id=None,
    )
    return updated, binding


__all__ = [
    "require_goal_creation",
    "require_current_goal",
    "create_goal_snapshot",
    "apply_goal_update",
    "admit_goal_snapshot",
]
