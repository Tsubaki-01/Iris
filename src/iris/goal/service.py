"""Goal 的同步领域入口，使用注入的同一生命周期存储。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from ..exceptions import IrisGoalStateError
from ..lifecycle.models import AgentRunOptions
from .config import GoalConfig
from .models import GoalReason, GoalRef, GoalRoundLimit, GoalSnapshot, GoalText
from .store import (
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    EditGoal,
    GoalStore,
    PauseGoal,
    ResumeGoal,
)

_CREATE_INPUT = TypeAdapter(tuple[GoalText, GoalText, GoalRoundLimit, AgentRunOptions])
_EDIT_INPUT = TypeAdapter(tuple[GoalText | None, GoalRoundLimit | None, AgentRunOptions | None])


class GoalService:
    """解析宿主创建/编辑输入并委托 store，不启动、恢复或取消 Run。"""

    def __init__(self, store: GoalStore, *, config: GoalConfig | None = None) -> None:
        self.store = store
        self.config = GoalConfig() if config is None else config

    def get_current(self, session_id: str) -> GoalSnapshot | None:
        """读取当前目标，保持只读。"""
        return self.store.get_current_goal(session_id)

    def get(self, goal_id: str) -> GoalSnapshot:
        """按 ID 读取目标，包括已取消当前选择的历史目标。"""
        return self.store.get_goal(goal_id)

    def create(
        self,
        session_id: str,
        objective: str,
        *,
        max_rounds: int | None = None,
        run_options: AgentRunOptions | None = None,
    ) -> GoalSnapshot:
        """创建目标并按需建立空 session，不占执行 lane。"""
        try:
            session_id, objective, rounds, options = _CREATE_INPUT.validate_python(
                (
                    session_id,
                    objective,
                    self.config.max_rounds if max_rounds is None else max_rounds,
                    AgentRunOptions() if run_options is None else run_options,
                )
            )
        except ValidationError as exc:
            raise IrisGoalStateError("目标创建参数无效", error=str(exc)) from exc
        return self.store.create_goal(
            CreateGoal(
                goal_id=f"goal_{uuid4().hex}",
                session_id=session_id,
                objective=objective,
                max_rounds=rounds,
                run_options=options,
                now=datetime.now(UTC),
            )
        )

    def edit(
        self,
        expected: GoalRef,
        *,
        objective: str | None = None,
        max_rounds: int | None = None,
        run_options: AgentRunOptions | None = None,
    ) -> GoalSnapshot:
        """修改显式提供的目标字段并暂停；不重置已用轮数。"""
        try:
            objective, max_rounds, run_options = _EDIT_INPUT.validate_python(
                (
                    objective,
                    max_rounds,
                    run_options,
                )
            )
        except ValidationError as exc:
            raise IrisGoalStateError("目标编辑参数无效", error=str(exc)) from exc
        return cast(
            GoalSnapshot,
            self.store.update_goal(
                EditGoal(
                    expected=expected,
                    objective=objective,
                    max_rounds=max_rounds,
                    run_options=run_options,
                    now=datetime.now(UTC),
                )
            ),
        )

    def pause(self, expected: GoalRef, *, reason: GoalReason) -> GoalSnapshot:
        """暂停目标状态；正在执行的 Run 由宿主独立处理。"""
        return cast(
            GoalSnapshot,
            self.store.update_goal(
                PauseGoal(
                    expected=expected,
                    reason=reason,
                    now=datetime.now(UTC),
                )
            ),
        )

    def resume(self, expected: GoalRef) -> GoalSnapshot:
        """恢复持久状态；已有在途 Run 是否可恢复由 store 判断。"""
        return cast(
            GoalSnapshot,
            self.store.update_goal(
                ResumeGoal(
                    expected=expected,
                    now=datetime.now(UTC),
                )
            ),
        )

    def complete(self, expected: GoalRef, *, reason: GoalReason) -> GoalSnapshot:
        """宿主显式完成目标，不自动结束正在执行的 Run。"""
        return cast(
            GoalSnapshot,
            self.store.update_goal(
                CompleteGoal(
                    expected=expected,
                    reason=reason,
                    now=datetime.now(UTC),
                )
            ),
        )

    def clear(self, session_id: str, *, expected: GoalRef | None) -> GoalSnapshot | None:
        """取消当前选择，保留目标和 Run 绑定供历史查询。"""
        return self.store.update_goal(
            ClearGoal(
                session_id=session_id,
                expected=expected,
                now=datetime.now(UTC),
            )
        )


__all__ = ["GoalService"]
