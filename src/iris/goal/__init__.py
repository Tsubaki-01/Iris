"""跨 Run 目标的领域模型、状态服务与原子存储契约。"""

from .config import GoalConfig
from .models import (
    GoalAdmission,
    GoalDecision,
    GoalProcessState,
    GoalReason,
    GoalRef,
    GoalReport,
    GoalRunBinding,
    GoalSnapshot,
    GoalStatus,
    GoalView,
)
from .service import GoalService
from .store import (
    AdmitGoalRun,
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    EditGoal,
    GoalStore,
    GoalUpdate,
    PauseGoal,
    ResumeGoal,
)

__all__ = [
    "GoalService",
    "GoalConfig",
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
