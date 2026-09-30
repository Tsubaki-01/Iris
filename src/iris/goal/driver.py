"""Goal 的纯续跑策略和唯一候选，不拥有执行任务或 memory 计数。"""

from ..exceptions import IrisGoalStateError
from .models import GoalContinuationIntent, GoalSnapshot, GoalStatus


class GoalDriver:
    """记录当前进程的推进意图，宿主负责准入、执行与候选清理。"""

    def __init__(self) -> None:
        """新控制器总是 disarmed，不能从持久 active 状态隐式启动。"""
        self._armed_goal_id: str | None = None
        self._intent: GoalContinuationIntent | None = None

    @property
    def armed_goal_id(self) -> str | None:
        """返回当前被显式允许推进的目标身份。"""
        return self._armed_goal_id

    @property
    def intent(self) -> GoalContinuationIntent | None:
        """返回尚未准入的唯一候选。"""
        return self._intent

    def arm(self, goal_id: str) -> None:
        """允许推进目标，持久状态和执行准入仍由各自 owner 决定。"""
        self._armed_goal_id = goal_id

    def disarm(self) -> GoalContinuationIntent | None:
        """停止推进并返回被撤销候选，供宿主释放其预留。"""
        self._armed_goal_id = None
        return self.invalidate()

    def can_continue(self, goal: GoalSnapshot | None) -> bool:
        """只判断目标策略；用户输入、lane、容量由 manager 统一判断。"""
        return (
            goal is not None
            and self._armed_goal_id == goal.goal_id
            and goal.status is GoalStatus.ACTIVE
            and goal.rounds_started < goal.max_rounds
        )

    def offer(
        self,
        source_run_id: str | None,
        goal_id: str,
        run_id: str,
    ) -> GoalContinuationIntent:
        """建立可撤销候选；相同终态重复到达时复用原候选身份。"""
        if self._intent is not None:
            if self._intent.source_run_id != source_run_id or self._intent.goal_id != goal_id:
                raise IrisGoalStateError("替换续跑候选前必须先清理原候选")
            return self._intent
        self._intent = GoalContinuationIntent(
            source_run_id=source_run_id,
            goal_id=goal_id,
            run_id=run_id,
        )
        return self._intent

    def consume(self) -> GoalContinuationIntent | None:
        """取出候选；准入后的清理责任交给持有其 run_id 的启动操作。"""
        return self.invalidate()

    def invalidate(self) -> GoalContinuationIntent | None:
        """移除并返回候选，让宿主使用同一身份执行幂等清理。"""
        previous, self._intent = self._intent, None
        return previous


__all__ = ["GoalDriver"]
