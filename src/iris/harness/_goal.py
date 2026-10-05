"""Goal 与单会话准入、恢复及资源交接的薄连接。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, cast

from ..agents import AgentConfig
from ..exceptions import IrisConfigError, IrisGoalConflictError, IrisGoalStateError
from ..goal.driver import GoalDriver
from ..goal.models import (
    GoalAdmission,
    GoalChanged,
    GoalContinuationIntent,
    GoalControlDisposition,
    GoalControlResult,
    GoalCreateInput,
    GoalEditInput,
    GoalProcessState,
    GoalReason,
    GoalSnapshot,
    GoalStatus,
    GoalView,
)
from ..goal.service import GoalService
from ..goal.session import GoalSession
from ..hitl import InteractionStatus
from ..lifecycle import AgentRunOptions, RunErrorInfo, RunEvent, RunEventKind, RunPhase, RunResult
from ..runtime import RuntimeCursor
from ..runtime.runtime import _normalize_run_error

if TYPE_CHECKING:
    from .session_manager import SessionManager
    from .streaming import CommandCleanupFailed

logger = logging.getLogger(__name__)


def validate_goal_options(config: AgentConfig, options: AgentRunOptions) -> None:
    """在创建、编辑或模型恢复入口检查最终工具策略。"""
    choice = options.runtime.request_options.get("tool_choice", config.model.tool_choice)
    if not options.runtime.include_tools or choice not in (None, "auto"):
        raise IrisConfigError(
            "Goal 自动运行需要 include_tools=true 且 tool_choice 为 auto 或未指定"
        )


class _GoalControl:
    """由 manager 持有；只在其锁内准入，异步准备留在锁外。"""

    def __init__(self, manager: SessionManager, service: GoalService) -> None:
        self.manager = manager
        self.runner = manager._runner
        self.service = service
        self.driver = GoalDriver()
        self.error: RunErrorInfo | None = None
        self._pending_error: RunErrorInfo | None = None
        self._operations: set[asyncio.Task[None]] = set()
        self._intent_task: asyncio.Task[None] | None = None
        self._pending_resume: tuple[str, str | None] | None = None
        self._resume_arming: asyncio.Task[RunResult] | None = None
        self._fact_callback = self.on_fact
        self.runner._register_session_fact_callback(manager._session_id, self._fact_callback)
        self.runner._goal_state_readers[manager._session_id] = self.process_state
        self.api = GoalSession(service, self)

    def process_state(self) -> GoalProcessState:
        """提供只读进程状态，不启动或结算执行。"""
        return GoalProcessState(armed_goal_id=self.driver.armed_goal_id, error=self.error)

    async def get(self) -> GoalView:
        """读取统一视图，不推进状态。"""
        async with self.manager._lock:
            self.manager._require_open()
            return self.service.get_view(self.manager._session_id)

    async def create(self, input_data: GoalCreateInput) -> GoalControlResult:
        """保存显式目标后允许本 manager 自动推进。"""
        async with self.manager._lock:
            self.manager._require_open()
            await self.manager._reconcile_locked()
            goal = self.service.create_validated(self.manager._session_id, input_data)
            self.driver.arm(goal.goal_id)
            self.error = None
            self.schedule_locked()
            return self._result("scheduled")

    async def edit(self, input_data: GoalEditInput) -> GoalControlResult:
        """在同一准入锁内读取版本并编辑，暂停后续轮次。"""
        async with self.manager._lock:
            self.manager._require_open()
            self.disarm()
            try:
                self.reconcile_locked()
                self.service.edit_validated(self._current().ref, input_data)
                self.error = None
            except Exception as exc:
                self.fail(exc)
                raise
            return self._result("stopped")

    async def pause(self, reason: GoalReason) -> GoalControlResult:
        """暂停目标，不取消已准入 Run。"""
        return await self._stop(reason, complete=False)

    async def complete(self, reason: GoalReason) -> GoalControlResult:
        """接受用户完成决定，不取消已准入 Run。"""
        return await self._stop(reason, complete=True)

    async def _stop(self, reason: GoalReason, *, complete: bool) -> GoalControlResult:
        async with self.manager._lock:
            self.manager._require_open()
            self.disarm()
            try:
                self.reconcile_locked()
                goal = self._current()
                if complete:
                    self.service.complete(goal.ref, reason=reason)
                else:
                    self.service.pause(goal.ref, reason=reason)
                self.error = None
            except Exception as exc:
                self.fail(exc)
                raise
            return self._result("stopped")

    async def clear(self) -> GoalControlResult:
        """清除当前选择，保留旧 Run 及其绑定。"""
        async with self.manager._lock:
            self.manager._require_open()
            self.disarm()
            try:
                self.reconcile_locked()
                current = self.service.get_current(self.manager._session_id)
                self.service.clear(
                    self.manager._session_id, expected=None if current is None else current.ref
                )
                self.error = None
            except Exception as exc:
                self.fail(exc)
                raise
            return self._result("stopped")

    async def resume(self, *, expected_activation_id: str | None = None) -> GoalControlResult:
        """显式接手原 Run；空闲时才允许创建下一轮。"""
        async with self.manager._lock:
            self.manager._require_open()
            abnormal = self.reconcile_locked()
            await self.manager._reconcile_locked()
            goal = self._current()
            if abnormal or goal.status is GoalStatus.COMPLETED:
                self.disarm()
                return self._result("stopped")
            run_id = self.service.store.load_session_lane(self.manager._session_id)
            if run_id is None:
                goal = self.service.resume(goal.ref)
                self.error = None
                if goal.status is GoalStatus.ACTIVE:
                    self.driver.arm(goal.goal_id)
                    self.schedule_locked()
                    return self._result("scheduled")
                self.disarm()
                return self._result("stopped")
            binding = self.service.store.get_goal_run(run_id)
            if binding is None or binding.goal_id != goal.goal_id:
                self.disarm()
                return self._result("occupied")
            run = self.runner.store.load_run(run_id)
            assert run is not None
            task = self.manager._current_task
            if self.manager._current_run_id == run_id and task is not None and not task.done():
                self.service.resume(goal.ref)
                if run_id not in self.runner._active and run.phase is RunPhase.ACTIVE:
                    self._resume_arming = task
                    return self._result("scheduled")
                self.driver.arm(goal.goal_id)
                self.error = None
                return self._result("waiting" if run.phase is RunPhase.WAITING else "running")
            cleanup_pending = run_id in self.runner._command_lifecycle.pending
            if (
                run.phase is RunPhase.ACTIVE
                and expected_activation_id is None
                and not cleanup_pending
            ):
                self.disarm()
                return self._result("needs_recovery")
            buffer = self.manager._event_buffer
            if buffer is not None and run_id not in buffer._run_trackers:
                if not buffer.try_register_run(run_id, after_sequence=run.last_event_sequence):
                    self.disarm()
                    self._pending_resume = (goal.goal_id, expected_activation_id)
                    return self._result("scheduled")
            self.manager._current_run_id = run_id
            self.runner._command_lifecycle.register_deadline(run, self.runner._command_target)
            view = self.service.get_view(self.manager._session_id)
            now = self.runner._now()
            interaction = view.interaction
            deadline = run.options.limits.deadline_at
            pending_answer = (
                run.phase is RunPhase.WAITING
                and interaction is not None
                and interaction.status is InteractionStatus.PENDING
                and (interaction.expires_at is None or interaction.expires_at > now)
                and (deadline is None or deadline > now)
                and run.cancellation_requested_at is None
                and not cleanup_pending
            )
            if pending_answer:
                self.service.resume(goal.ref)
                self.driver.arm(goal.goal_id)
                self.error = None
                return self._result("waiting")
            self.disarm()
            self.service.resume(goal.ref)
            started = asyncio.Event()
            task = asyncio.create_task(
                self.runner._recover_managed(
                    run_id,
                    expected_activation_id=expected_activation_id,
                    steering=self.manager._steering,
                    durable_event_callback=self.manager._relay_run_event,
                    activation_started=started,
                )
            )
            self.manager._current_task = task
            self.manager._attach_settlement_callback(task, run_id, submission=None)
            self._resume_arming = task
            goal_id = goal.goal_id
        # 原 Run 已有持久身份；等待准备/注册期间仍允许 pause、interrupt 与 close 取锁。
        try:
            await self.manager._wait_for_admission(task, started)
        except Exception as exc:
            async with self.manager._lock:
                self.fail(exc)
            raise
        async with self.manager._lock:
            if not self._finish_resume_arming(task, goal_id):
                return self._result("stopped")
            await self.manager._reconcile_locked()
            self.schedule_locked()
            view = self.service.get_view(self.manager._session_id)
            if view.run is not None:
                return self._result("waiting" if view.run.phase is RunPhase.WAITING else "running")
            return self._result("scheduled")

    def _finish_resume_arming(self, task: asyncio.Task[RunResult], goal_id: str) -> bool:
        """接手后只完成仍有效的 resume 意图；结算和故障可使该意图失效。"""
        abnormal = self.reconcile_locked()
        may_arm = self._resume_arming is task and not self.manager._closed
        if self._resume_arming is task:
            self._resume_arming = None
        goal = self.service.get_current(self.manager._session_id)
        if (
            not may_arm
            or abnormal
            or goal is None
            or goal.goal_id != goal_id
            or goal.status is not GoalStatus.ACTIVE
        ):
            return False
        self.driver.arm(goal_id)
        self.error = None
        return True

    def reserve_successor(self, source_run_id: str) -> None:
        """旧调用退出前同步暂存候选，不读写存储或启动任务。"""
        goal_id = self.driver.armed_goal_id
        if goal_id is not None:
            intent = self.driver.intent or self.driver.offer(
                source_run_id, goal_id, self.manager._new_run_id()
            )
            self.manager._reserve_maintenance_handoff(intent.run_id)

    def cancel_candidate(self) -> None:
        """仅取消本 Goal 尚未准入的意图和预留。"""
        intent = self.driver.invalidate()
        if intent is not None:
            self.manager._release_maintenance_handoff(intent.run_id)
            if self.manager._event_buffer is not None:
                self.manager._event_buffer.discard_run(intent.run_id)
            task = self._intent_task
            self._intent_task = None
            if task is not None and task is not asyncio.current_task():
                task.cancel()

    def disarm(self) -> None:
        """停止未来轮次，不撤销已准入 Run。"""
        self.cancel_candidate()
        self.driver.disarm()
        self._pending_resume = None
        self._resume_arming = None

    def reconcile_locked(self) -> bool:
        """补结算已提交终态；失败仅停止 Goal，普通用户仍可准入。"""
        if self._pending_error is not None:
            self.disarm()
            self.error, self._pending_error = self._pending_error, None
            self.publish()
        try:
            settlements = self.service.reconcile(self.manager._session_id)
            current = self.service.get_current(self.manager._session_id)
            abnormal = False
            for settled in settlements:
                if current is not None and settled.binding.goal_id == current.goal_id:
                    result = self.runner.get_result(settled.binding.run_id)
                    abnormal |= result is not None and result.run.stop_reason != "completed"
            if current is None or current.status is not GoalStatus.ACTIVE:
                self.disarm()
            if settlements:
                self.publish()
            return abnormal
        except Exception as exc:
            self.fail(exc)
            return True

    def schedule_locked(self) -> None:
        """只排一个可撤销操作；真正准备和准入另一次竞争 manager lock。"""
        manager = self.manager
        if manager._closed:
            self.cancel_candidate()
            return
        if self._pending_resume is not None:
            goal_id, expected = self._pending_resume
            self._pending_resume = None
            if self._current().goal_id == goal_id:
                self._track_operation(self._retry_resume(expected))
            return
        error = self.runner._hook_lifecycle.admission_error
        if error is not None:
            self.fail(error)
            return
        goal = self.service.get_current(manager._session_id)
        if (
            not self.driver.can_continue(goal)
            or manager._current_run_id is not None
            or manager._pending.peek_follow_up() is not None
            or self.service.store.load_session_lane(manager._session_id) is not None
        ):
            self.cancel_candidate()
            return
        if any(not task.done() for task in manager._managed_tasks):
            # 用户输入仍沿用既有 terminal 准入；自动轮等旧完整调用退出再接手。
            return
        if self.runner._hook_lifecycle.has_pending(manager._session_id):
            return
        if self._intent_task is not None and not self._intent_task.done():
            return
        assert goal is not None
        intent = self.driver.intent or self.driver.offer(None, goal.goal_id, manager._new_run_id())
        if manager._event_buffer is not None and not manager._event_buffer.try_register_run(
            intent.run_id, after_sequence=0
        ):
            self.cancel_candidate()
            return
        self._intent_task = self._track_operation(self._start_intent(intent))

    def _track_operation(self, operation: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(operation)
        self._operations.add(task)
        task.add_done_callback(self._operation_finished)
        return task

    def _operation_finished(self, task: asyncio.Task[None]) -> None:
        self._operations.discard(task)
        if not task.cancelled():
            error = task.exception()
            if isinstance(error, Exception):
                self.fail(error)

    async def _retry_resume(self, expected: str | None) -> None:
        try:
            await self.resume(expected_activation_id=expected)
        except Exception as exc:
            self.fail(exc)

    async def _start_intent(self, intent: GoalContinuationIntent) -> None:
        manager = self.manager
        admitted = False
        execution: asyncio.Task[RunResult | None] | None = None

        async def admit() -> tuple[GoalAdmission, RuntimeCursor] | None:
            nonlocal admitted
            async with manager._lock:
                goal = self.service.get_current(manager._session_id)
                if (
                    manager._closed
                    or self.driver.intent is not intent
                    or not self.driver.can_continue(goal)
                    or manager._current_run_id is not None
                    or manager._pending.peek_follow_up() is not None
                    or self.service.store.load_session_lane(manager._session_id) is not None
                    or self.runner._hook_lifecycle.has_pending(manager._session_id)
                ):
                    return None
                assert goal is not None and execution is not None
                created = self.runner._admit_goal_start(goal.ref, run_id=intent.run_id)
                admitted = True
                self.driver.consume()
                self._intent_task = None
                manager._current_run_id = intent.run_id
                task = cast(asyncio.Task[RunResult], execution)
                manager._current_task = task
                manager._attach_settlement_callback(task, intent.run_id, submission=None)
                self.publish()
                return created

        try:
            await asyncio.sleep(0)
            goal = self.service.get(intent.goal_id)
            started = asyncio.Event()
            execution = asyncio.create_task(
                self.runner._start_goal_managed(
                    goal.ref,
                    run_id=intent.run_id,
                    admit=admit,
                    steering=manager._steering,
                    durable_event_callback=manager._relay_run_event,
                    activation_started=started,
                )
            )
            await manager._wait_for_admission(cast(asyncio.Task[RunResult], execution), started)
            async with manager._lock:
                if self._resume_arming is execution and self._finish_resume_arming(
                    cast(asyncio.Task[RunResult], execution), intent.goal_id
                ):
                    await manager._reconcile_locked()
                    self.schedule_locked()
                    self.publish()
        except asyncio.CancelledError:
            raise
        except IrisGoalConflictError:
            async with manager._lock:
                self.cancel_candidate()
                self.reconcile_locked()
                self.publish()
        except Exception as exc:
            async with manager._lock:
                if not admitted and self.driver.intent is intent:
                    current = self.service.get_current(manager._session_id)
                    if current is not None and current.goal_id == intent.goal_id:
                        try:
                            self.service.pause(
                                current.ref,
                                reason=GoalReason(
                                    code="start_failed",
                                    text=str(exc),
                                ),
                            )
                        except Exception:
                            logger.exception("Goal 启动失败后无法保存暂停状态")
                self.fail(exc)
        finally:
            manager._release_maintenance_handoff(intent.run_id)
            if not admitted:
                if execution is not None:
                    if not execution.done():
                        execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                if self.driver.intent is intent:
                    self.cancel_candidate()

    def on_fact(self, fact: RunEvent | CommandCleanupFailed) -> None:
        """同步事实回调只登记水位/错误，并合并唤醒锁内 reconciliation。"""
        if self.manager._closed:
            return
        if isinstance(fact, RunEvent):
            if fact.kind is not RunEventKind.RUN_TERMINAL:
                return
            if self.manager._event_buffer is not None:
                self.manager._event_buffer.observe_run_event(fact)
        else:
            binding = self.service.store.get_goal_run(fact.run_id)
            if binding is None:
                return
            self._pending_error = fact.error
        self.manager._schedule_tracker_reconcile()

    def execution_finished(self, run_id: str, error: Exception | None) -> None:
        """实际 invocation 退出后显示等待/故障；不自动恢复异常 activation。"""
        if error is not None and self.service.store.get_goal_run(run_id) is not None:
            self.fail(error)
        else:
            self.publish()

    def interrupt_locked(self, reason: str | None) -> bool:
        """暂停未完成目标；Run 的实际取消仍归 SessionManager。"""
        was_armed = self.driver.armed_goal_id is not None
        self.disarm()
        goal = self.service.get_current(self.manager._session_id)
        if goal is None or goal.status is GoalStatus.COMPLETED:
            return False
        was_active = goal.status is not GoalStatus.PAUSED
        if goal.status is not GoalStatus.PAUSED:
            self.service.pause(
                goal.ref,
                reason=GoalReason(
                    code="user_interrupt",
                    text=reason or "用户中断目标",
                ),
            )
        self.publish()
        return was_active or was_armed

    def close_locked(self) -> None:
        """注销 manager 附着，取消尚未准入的等待，不关闭 Runner 资源。"""
        self.disarm()
        self.runner._unregister_session_fact_callback(self.manager._session_id, self._fact_callback)
        self.runner._goal_state_readers.pop(self.manager._session_id, None)

    async def wait_closed(self) -> None:
        """等待本接入层的短操作收尾，不等待已分离的完整 Run。"""
        await asyncio.gather(*tuple(self._operations), return_exceptions=True)

    def fail(self, error: Exception) -> None:
        """保存真实进程错误并停止自动调度。"""
        self.disarm()
        self.error = _normalize_run_error(error)
        self.publish()

    def publish(self) -> None:
        """通知只投影真实状态，失败不回滚状态或驱动下一轮。"""
        try:
            self.manager._publish_goal_changed(
                GoalChanged(
                    session_id=self.manager._session_id,
                    view=self.service.get_view(self.manager._session_id),
                )
            )
        except Exception:
            logger.exception("Goal 状态通知失败", extra={"session_id": self.manager._session_id})

    def _current(self) -> GoalSnapshot:
        goal = self.service.get_current(self.manager._session_id)
        if goal is None:
            raise IrisGoalStateError("当前 session 没有 Goal")
        return goal

    def _result(self, disposition: GoalControlDisposition) -> GoalControlResult:
        self.publish()
        return GoalControlResult(
            view=self.service.get_view(self.manager._session_id),
            disposition=disposition,
        )
