"""Goal 的同步领域入口，使用注入的同一生命周期存储。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from ..exceptions import IrisGoalConflictError, IrisGoalStateError
from ..lifecycle.models import AgentRunOptions, RunPhase, snapshot_run
from .config import GoalConfig
from .models import (
    GoalCreateInput,
    GoalEditInput,
    GoalProcessState,
    GoalReason,
    GoalRef,
    GoalReport,
    GoalRoundLimit,
    GoalSnapshot,
    GoalStatus,
    GoalText,
    GoalView,
)
from .store import (
    ClearGoal,
    CompleteGoal,
    CreateGoal,
    EditGoal,
    GoalStore,
    PauseGoal,
    ResumeGoal,
)
from .transitions import require_current_goal

if TYPE_CHECKING:
    from .models import GoalSettlement

_SESSION_ID_INPUT = TypeAdapter(GoalText)
_CREATE_INPUT = TypeAdapter(tuple[GoalText, GoalRoundLimit, AgentRunOptions])
_EDIT_INPUT = TypeAdapter(tuple[GoalText | None, GoalRoundLimit | None, AgentRunOptions | None])


def _empty_process_state(session_id: str) -> GoalProcessState:
    """没有附着控制器时不推断进程内运行意图。"""
    return GoalProcessState()


class GoalService:
    """解析宿主创建/编辑输入并委托 store，不启动、恢复或取消 Run。"""

    def __init__(
        self,
        store: GoalStore,
        *,
        config: GoalConfig | None = None,
        process_state_reader: Callable[[str], GoalProcessState] | None = None,
        run_options_validator: Callable[[AgentRunOptions], None] | None = None,
    ) -> None:
        """注入同一存储与可选宿主只读能力，不依赖宿主实现。"""
        self.store = store
        self.config = GoalConfig() if config is None else config
        self.process_state_reader = process_state_reader or _empty_process_state
        self.run_options_validator = run_options_validator

    def get_current(self, session_id: str) -> GoalSnapshot | None:
        """读取当前目标，保持只读。"""
        return self.store.get_current_goal(session_id)

    def get(self, goal_id: str) -> GoalSnapshot:
        """按 ID 读取目标，包括已取消当前选择的历史目标。"""
        return self.store.get_goal(goal_id)

    def get_view(self, session_id: str) -> GoalView:
        """组合当前目标、占用 Run、人工交互和待结算事实，保持只读。"""
        goal = self.store.get_current_goal(session_id)
        process = self.process_state_reader(session_id)
        run_id = self.store.load_session_lane(session_id)
        run = None if run_id is None else self.store.load_run(run_id)
        settlement_pending = False
        for binding in self.store.list_unsettled_goal_runs(session_id):
            pending_run = self.store.load_run(binding.run_id)
            if pending_run is not None and pending_run.phase is RunPhase.TERMINAL:
                settlement_pending = True
        run_binding = None if run is None else self.store.get_goal_run(run.run_id)
        interaction = (
            None
            if run is None or run.pending_interaction_id is None
            else self.store.load_interaction(run.pending_interaction_id)
        )
        return GoalView(
            goal=goal,
            armed=(
                goal is not None
                and goal.status is GoalStatus.ACTIVE
                and process.armed_goal_id == goal.goal_id
            ),
            run=None if run is None else snapshot_run(run),
            run_goal_id=None if run_binding is None else run_binding.goal_id,
            interaction=interaction,
            settlement_pending=settlement_pending,
            driver_error=process.error,
        )

    def report(self, run_id: str, report: GoalReport) -> GoalReport:
        """核对当前绑定与版本后原样返回申报，不修改目标或绑定。"""
        binding = self.store.get_goal_run(run_id)
        if binding is None:
            raise IrisGoalStateError("本轮未绑定 Goal，不能申报目标结果", run_id=run_id)
        if binding.goal_id != report.goal_id:
            raise IrisGoalConflictError("本轮绑定的目标与申报不一致", run_id=run_id)
        bound_goal = self.store.get_goal(binding.goal_id)
        current = self.store.get_current_goal(bound_goal.session_id)
        if current is None:
            raise IrisGoalConflictError("本轮目标已清除，不能申报", goal_id=report.goal_id)
        require_current_goal(
            current,
            GoalRef.model_construct(goal_id=report.goal_id, revision=report.revision),
            is_current=True,
        )
        return report

    def settle_run(self, run_id: str, *, now: datetime) -> GoalSettlement:
        """委托存储基于同一事务内的运行事实结算一次绑定。"""
        return self.store.settle_goal_run(run_id, now=now)

    def reconcile(self, session_id: str) -> tuple[GoalSettlement, ...]:
        """显式补结算本 session 已终态的绑定，不接管在途 Run。"""
        results: list[GoalSettlement] = []
        for binding in self.store.list_unsettled_goal_runs(session_id):
            run = self.store.load_run(binding.run_id)
            if run is not None and run.phase is RunPhase.TERMINAL:
                results.append(self.settle_run(binding.run_id, now=datetime.now(UTC)))
        return tuple(results)

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
            session_id = _SESSION_ID_INPUT.validate_python(session_id)
        except ValidationError as exc:
            raise IrisGoalStateError("目标 session ID 无效", error=str(exc)) from exc
        return self.create_validated(
            session_id,
            self.parse_create(objective, max_rounds=max_rounds, run_options=run_options),
        )

    def parse_create(
        self,
        objective: str,
        *,
        max_rounds: int | None = None,
        run_options: AgentRunOptions | None = None,
    ) -> GoalCreateInput:
        """一次解析创建字段，供同步服务和异步会话 SDK 共用。"""
        try:
            objective, rounds, options = _CREATE_INPUT.validate_python(
                (
                    objective,
                    self.config.max_rounds if max_rounds is None else max_rounds,
                    AgentRunOptions() if run_options is None else run_options,
                )
            )
        except ValidationError as exc:
            raise IrisGoalStateError("目标创建参数无效", error=str(exc)) from exc
        if self.run_options_validator is not None:
            self.run_options_validator(options)
        return GoalCreateInput(objective=objective, max_rounds=rounds, run_options=options)

    def create_validated(self, session_id: str, input_data: GoalCreateInput) -> GoalSnapshot:
        """保存已解析的输入；宿主在 admission lock 内调用，不重新解析。"""
        return self.store.create_goal(
            CreateGoal(
                goal_id=f"goal_{uuid4().hex}",
                session_id=session_id,
                objective=input_data.objective,
                max_rounds=input_data.max_rounds,
                run_options=input_data.run_options,
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
        return self.edit_validated(
            expected,
            self.parse_edit(objective=objective, max_rounds=max_rounds, run_options=run_options),
        )

    def parse_edit(
        self,
        *,
        objective: str | None = None,
        max_rounds: int | None = None,
        run_options: AgentRunOptions | None = None,
    ) -> GoalEditInput:
        """一次解析非空编辑，内部字段与模型配置约束不在写入路径重验。"""
        if objective is None and max_rounds is None and run_options is None:
            raise IrisGoalStateError("编辑目标至少提供一个修改字段")
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
        if run_options is not None and self.run_options_validator is not None:
            self.run_options_validator(run_options)
        return GoalEditInput(objective=objective, max_rounds=max_rounds, run_options=run_options)

    def edit_validated(self, expected: GoalRef, input_data: GoalEditInput) -> GoalSnapshot:
        """在锁内取得当前引用后应用可信编辑，仅由 store 检查状态和 CAS。"""
        return cast(
            GoalSnapshot,
            self.store.update_goal(
                EditGoal(
                    expected=expected,
                    objective=input_data.objective,
                    max_rounds=input_data.max_rounds,
                    run_options=input_data.run_options,
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
