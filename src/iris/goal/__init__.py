"""跨 Run 目标的领域模型、状态服务与原子存储契约。"""

from .config import GoalConfig
from .models import (
    GoalChanged,
    GoalControlDisposition,
    GoalControlResult,
    GoalDecision,
    GoalProcessState,
    GoalReason,
    GoalRef,
    GoalReport,
    GoalRunBinding,
    GoalSettlement,
    GoalSnapshot,
    GoalStatus,
    GoalView,
)
from .service import GoalService
from .session import GoalSession
from .store import GoalStore

__all__ = [
    "GoalChanged",
    "GoalControlDisposition",
    "GoalControlResult",
    "GoalSession",
    "GoalService",
    "GoalConfig",
    "GoalStatus",
    "GoalDecision",
    "GoalRef",
    "GoalReason",
    "GoalReport",
    "GoalRunBinding",
    "GoalSettlement",
    "GoalSnapshot",
    "GoalProcessState",
    "GoalView",
    "GoalStore",
]
