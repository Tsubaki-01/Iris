"""Goal 持久事实和进程内只读投影，不持有执行任务。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from ..hitl.models import HumanInteraction
from ..lifecycle.models import AgentRunOptions, RunErrorInfo, RunSnapshot
from ..lifecycle.store import RunCommit

GoalText = Annotated[str, Field(pattern=r"\S")]
GoalRoundLimit = Annotated[int, Field(gt=0)]


class GoalStatus(StrEnum):
    """目标的四种持久业务状态。"""

    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class GoalDecision(StrEnum):
    """模型可以申报的结果，不含宿主控制操作。"""

    COMPLETE = "complete"
    BLOCKED = "blocked"
    CONTINUE = "continue"


class _GoalModel(BaseModel):
    """目标外部解析和持久数据的统一不可变模型。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GoalRef(_GoalModel):
    """目标身份及精确版本，不复用 session history revision。"""

    goal_id: GoalText
    revision: int = Field(ge=0)


class GoalReason(_GoalModel):
    """供宿主解释状态变化的稳定原因码与正文。"""

    code: GoalText
    text: GoalText


class GoalReport(_GoalModel):
    """模型申报；run 与 call 身份由真实执行上下文提供。"""

    goal_id: GoalText
    revision: int = Field(ge=0)
    decision: GoalDecision
    reason: GoalText


class GoalRunBinding(_GoalModel):
    """目标与自动顶层 Run 的持久绑定及结算位置。"""

    run_id: GoalText
    goal_id: GoalText
    round_no: int = Field(gt=0)
    admission_revision: int = Field(ge=0)
    settled_at: AwareDatetime | None = None
    applied_report_call_id: GoalText | None = None

    @model_validator(mode="after")
    def _validate_settlement(self) -> GoalRunBinding:
        if self.applied_report_call_id is not None and self.settled_at is None:
            raise ValueError("已应用报告的绑定必须已结算")
        return self


class GoalSnapshot(_GoalModel):
    """目标权威快照；不复制 Run 状态、使用量或进程 armed 标记。"""

    goal_id: GoalText
    session_id: GoalText
    revision: int = Field(ge=0)
    objective: GoalText
    status: GoalStatus
    reason: GoalReason | None = None
    max_rounds: GoalRoundLimit
    rounds_started: int = Field(ge=0)
    run_options: AgentRunOptions
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def _validate_rounds(self) -> GoalSnapshot:
        if self.rounds_started > self.max_rounds:
            raise ValueError("已使用轮数不能超过目标总额度")
        return self

    @property
    def ref(self) -> GoalRef:
        """投影当前精确版本，不重新解析已验证字段。"""
        return GoalRef.model_construct(goal_id=self.goal_id, revision=self.revision)


@dataclass(frozen=True, slots=True, kw_only=True)
class GoalAdmission:
    """原子准入的完整回执，不创建第二份 lifecycle 事实。"""

    commit: RunCommit
    goal: GoalSnapshot
    binding: GoalRunBinding


@dataclass(frozen=True, slots=True, kw_only=True)
class GoalProcessState:
    """宿主提供的只读进程状态；未附着时为空。"""

    armed_goal_id: str | None = None
    error: RunErrorInfo | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GoalView:
    """持久事实与进程状态的统一只读投影。"""

    goal: GoalSnapshot | None
    armed: bool
    run: RunSnapshot | None
    run_goal_id: str | None
    interaction: HumanInteraction | None
    settlement_pending: bool
    driver_error: RunErrorInfo | None


__all__ = [
    "GoalStatus",
    "GoalDecision",
    "GoalRef",
    "GoalReason",
    "GoalReport",
    "GoalRunBinding",
    "GoalSnapshot",
    "GoalAdmission",
    "GoalProcessState",
    "GoalView",
]
