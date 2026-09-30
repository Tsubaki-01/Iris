"""Goal 的可信 command 与共享 lifecycle 事务存储协议。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..lifecycle.models import AgentRunOptions
from ..lifecycle.store import CreateRun, LifecycleStore
from .models import GoalAdmission, GoalReason, GoalRef, GoalRunBinding, GoalSettlement, GoalSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateGoal:
    """建立目标，ID 与时间由调用边界生成。"""

    goal_id: str
    session_id: str
    objective: str
    max_rounds: int
    run_options: AgentRunOptions
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class EditGoal:
    """修改明确提供的字段，保留已用轮数并暂停。"""

    expected: GoalRef
    now: datetime
    objective: str | None = None
    max_rounds: int | None = None
    run_options: AgentRunOptions | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PauseGoal:
    """暂停后续目标推进，不修改已存在 Run。"""

    expected: GoalRef
    reason: GoalReason
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeGoal:
    """恢复目标状态；实际附着与启动由 harness 决定。"""

    expected: GoalRef
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class CompleteGoal:
    """宿主明确完成目标，不修改已存在 Run。"""

    expected: GoalRef
    reason: GoalReason
    now: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearGoal:
    """取消当前目标选择，保留所有记录；空引用只匹配 absent。"""

    session_id: str
    expected: GoalRef | None
    now: datetime


GoalUpdate = EditGoal | PauseGoal | ResumeGoal | CompleteGoal | ClearGoal


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmitGoalRun:
    """将现有 CreateRun 与目标轮数、绑定放入同一事务。"""

    expected: GoalRef
    create_run: CreateRun


class GoalStore(LifecycleStore, Protocol):
    """同一个 backend 上的目标和生命周期权威存储。"""

    def get_current_goal(self, session_id: str) -> GoalSnapshot | None:
        """只读当前目标，无目标时返回 None。"""
        ...

    def get_goal(self, goal_id: str) -> GoalSnapshot:
        """按 ID 读取包括已清除的目标，不存在时抛领域异常。"""
        ...

    def get_goal_run(self, run_id: str) -> GoalRunBinding | None:
        """读取自动 Run 绑定；普通 Run 返回 None。"""
        ...

    def list_unsettled_goal_runs(self, session_id: str) -> tuple[GoalRunBinding, ...]:
        """读取本 session 所有未结算绑定，包括已 clear 的旧目标。"""
        ...

    def create_goal(self, command: CreateGoal) -> GoalSnapshot:
        """按需创建空 session 并原子建立目标。"""
        ...

    def update_goal(self, command: GoalUpdate) -> GoalSnapshot | None:
        """CAS 应用一次目标 mutation；clear absent 返回 None。"""
        ...

    def admit_goal_run(self, command: AdmitGoalRun) -> GoalAdmission:
        """原子创建普通 Run、绑定并消耗目标轮数。"""
        ...

    def settle_goal_run(self, run_id: str, *, now: datetime) -> GoalSettlement:
        """按同一快照中的终态运行证据，幂等结算目标与绑定。"""
        ...


__all__ = [
    "CreateGoal",
    "EditGoal",
    "PauseGoal",
    "ResumeGoal",
    "CompleteGoal",
    "ClearGoal",
    "GoalUpdate",
    "AdmitGoalRun",
    "GoalStore",
]
