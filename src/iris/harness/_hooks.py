"""Logical run Hook 投影及 root 共享的完成 owner；不写 durable 状态。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from ..command.models import CommandScope
from ..command.service import CommandBinding
from ..exceptions import IrisCommandCleanupError, IrisRunStateError
from ..hooks._dispatch_types import CommandHookRegistration, HookControl
from ..hooks.dispatcher import HookDispatcher
from ..hooks.models import RunFinishedEvent, RunStartedEvent
from ..lifecycle import RunRecord, RunResult, RunSnapshot, RunStopReason, snapshot_run

if TYPE_CHECKING:
    from ..runtime import RuntimeActivationInput
    from .runner import ActiveActivation, AgentRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HookStartControl:
    """START 的控制事实；runner 依当前 signal 映射既有结算意图。"""

    control: HookControl


async def run_started(
    runner: AgentRunner, active: ActiveActivation, activation: RuntimeActivationInput
) -> HookStartControl | None:
    """在已建立 active.task 的 START 内派发，不构造虚假 engine cursor。"""
    environment = runner.runtime.environment
    dispatcher = environment.hook_dispatcher
    if dispatcher is None:
        return None
    run = cast(RunRecord, runner.store.load_run(activation.run_id))
    outcome = await dispatcher.dispatch(
        RunStartedEvent(
            agent_id=run.agent_id,
            session_id=run.session_id,
            run_id=run.run_id,
            activation_id=activation.activation_id,
            workspace=str(environment.workspace_root),
            occurred_at=runner._now(),
            run=snapshot_run(run),
            input=run.request.input,
        ),
        cancellation=active.signal,
    )
    return None if outcome.control is None else HookStartControl(outcome.control)


@dataclass(slots=True)
class FinishedHook:
    """一个新终态 producer 登记的实际任务及其独立完成通知。"""

    run_id: str
    session_id: str
    activation_id: str | None
    workspace: str
    dispatcher: HookDispatcher
    binding: CommandBinding | None
    publication: asyncio.Future[RunResult | None]
    completion: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    started: bool = False
    cancel_requested: bool = False
    cancel_error: asyncio.CancelledError | None = None
    error: IrisCommandCleanupError | None = None


class HookLifecycle:
    """root 共享 finished owner 与新 Run 准入错误；child 仅借用本实例。"""

    def __init__(self, root: AgentRunner) -> None:
        """保留当前进程的任务与订阅，不恢复或补发历史 Hook。"""
        self.root = root
        self.admission_error: IrisCommandCleanupError | None = None
        self._finished: dict[str, FinishedHook] = {}
        self._subscribers: dict[str, set[Callable[[], None]]] = {}
        self._cancelling_sessions: set[str] = set()

    @contextmanager
    def cancelling_session(self, session_id: str) -> Iterator[None]:
        """Manager 的一次显式关闭也取消该期间新登记的完成 owner。"""
        self._cancelling_sessions.add(session_id)
        try:
            yield
        finally:
            self._cancelling_sessions.discard(session_id)

    def has_pending(self, session_id: str) -> bool:
        """查询该 session 是否尚有未完成的真实 owner。"""
        return any(
            entry.session_id == session_id and not entry.completion.is_set()
            for entry in self._finished.values()
        )

    def has_pending_run(self, run_id: str) -> bool:
        """供已终态驱动路径判断是否需要显式取消 owner。"""
        entry = self._finished.get(run_id)
        return entry is not None and not entry.completion.is_set()

    def check_admission(self, session_id: str) -> None:
        """只约束新 Run；原资源错误优先于同 session 的完成等待。"""
        if self.admission_error is not None:
            raise self.admission_error
        if self.has_pending(session_id):
            raise IrisRunStateError("session 的 Run 完成处理器尚未结束", session_id=session_id)

    def subscribe(self, session_id: str, callback: Callable[[], None]) -> None:
        """登记轻量唤醒；回调不拥有 Hook 任务。"""
        self._subscribers.setdefault(session_id, set()).add(callback)

    def unsubscribe(self, session_id: str, callback: Callable[[], None]) -> None:
        """只移除当前 attachment 的 exact 回调。"""
        subscribers = self._subscribers.get(session_id)
        if subscribers is not None:
            subscribers.discard(callback)
            if not subscribers:
                del self._subscribers[session_id]

    def register_finished(
        self,
        runner: AgentRunner,
        run: RunRecord | RunSnapshot,
        *,
        stop_reason: RunStopReason,
        activation_id: str | None = None,
    ) -> FinishedHook | None:
        """同步提交前登记唯一 owner；没有适用处理器时不改变现有并发。"""
        environment = runner.runtime.environment
        dispatcher = environment.hook_dispatcher
        if dispatcher is None or not any(
            registration.event == "run.finished"
            and (
                not isinstance(registration, CommandHookRegistration)
                or stop_reason is RunStopReason.COMPLETED
            )
            for registration in dispatcher.registrations
        ):
            return None
        if run.run_id in self._finished:
            return None
        entry = FinishedHook(
            run_id=run.run_id,
            session_id=run.session_id,
            activation_id=activation_id,
            workspace=str(environment.workspace_root),
            dispatcher=dispatcher,
            binding=environment.command_binding,
            publication=asyncio.get_running_loop().create_future(),
            cancel_requested=run.session_id in self._cancelling_sessions,
        )
        self._finished[run.run_id] = entry
        entry.task = asyncio.create_task(self._run_finished(entry))
        entry.task.add_done_callback(_retrieve_exception)
        return entry

    def withdraw_finished(self, entry: FinishedHook | None) -> None:
        """提交失败只撤销本 producer 的登记，不影响其他提交者。"""
        if entry is None or self._finished.get(entry.run_id) is not entry:
            return
        del self._finished[entry.run_id]
        if not entry.publication.done():
            entry.publication.set_result(None)

    def publish_finished(self, entry: FinishedHook | None, result: RunResult) -> None:
        """原 PendingSettlement 移除后发布新终态，允许已登记 owner 开始。"""
        if entry is not None and not entry.publication.done():
            entry.publication.set_result(result)

    async def wait_session(self, session_id: str) -> None:
        """普通等待者只等通知；取消它不会触及实际 owner，root 错误及时唤醒。"""
        changed = asyncio.Event()
        callback = changed.set
        self.subscribe(session_id, callback)
        try:
            while True:
                changed.clear()
                if self.admission_error is not None:
                    raise self.admission_error
                if not self.has_pending(session_id):
                    return
                await changed.wait()
        finally:
            self.unsubscribe(session_id, callback)

    async def join_finished(self, run_id: str) -> None:
        """实际驱动者等待 owner；驱动者取消明确中断并收口 owner 后再传播。"""
        entry = self._finished.get(run_id)
        if entry is None:
            return
        task = cast(asyncio.Task[None], entry.task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            await self.cancel_run(run_id)
            if entry.error is not None:
                raise entry.error from error
            raise
        if entry.error is not None:
            raise entry.error

    async def cancel_run(self, run_id: str) -> None:
        """显式取消实际 owner 一次；等待期间的重复取消不打断资源收口。"""
        entry = self._finished.get(run_id)
        if entry is None:
            return
        task = cast(asyncio.Task[None], entry.task)
        if not task.done() and not entry.cancel_requested:
            entry.cancel_requested = True
            if entry.started:
                task.cancel()
            elif not entry.publication.done():
                entry.publication.set_result(None)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        if entry.error is not None:
            raise entry.error

    async def cancel_session(self, session_id: str) -> None:
        """Manager 显式关闭该 session 时停止其实际完成任务。"""
        for entry in tuple(self._finished.values()):
            if entry.session_id == session_id and not entry.completion.is_set():
                await self.cancel_run(entry.run_id)

    async def aclose(self) -> None:
        """等待包括后台期限产生者在内的实际任务；历史错误不阻止环境关闭。"""
        tasks = [
            entry.task
            for entry in self._finished.values()
            if entry.task is not None and entry.task is not asyncio.current_task()
        ]
        if tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    async def _run_finished(self, entry: FinishedHook) -> None:
        """只操作新终态后的附加动作，永不再次 finish 或写 PendingSettlement。"""
        entry.started = True
        result: RunResult | None = None
        try:
            result = await entry.publication
            if result is None or entry.cancel_requested:
                return
            outcome = await entry.dispatcher.dispatch(
                RunFinishedEvent(
                    agent_id=result.run.agent_id,
                    session_id=entry.session_id,
                    run_id=entry.run_id,
                    activation_id=entry.activation_id,
                    workspace=entry.workspace,
                    occurred_at=self.root._now(),
                    result=result,
                ),
                cancellation=None,
            )
            if outcome.control is not None:
                if isinstance(outcome.control.error, asyncio.CancelledError):
                    entry.cancel_error = outcome.control.error
                await self._drain_finished(entry, outcome.control)
        except asyncio.CancelledError as error:
            entry.cancel_error = error
            if result is not None:
                try:
                    await self._drain_finished(entry, HookControl("task_cancelled", error))
                except IrisCommandCleanupError as cleanup_error:
                    self._record_failure(entry, cleanup_error)
        except IrisCommandCleanupError as error:
            self._record_failure(entry, error)
        finally:
            entry.completion.set()
            if entry.error is None and self._finished.get(entry.run_id) is entry:
                del self._finished[entry.run_id]
            self._notify(None if entry.error is not None else entry.session_id)

    async def _drain_finished(self, entry: FinishedHook, control: HookControl) -> None:
        """后终态只收口现有命令资源；失败锁 root 准入，不重跑脚本。"""
        binding = entry.binding
        slot = control.stop_slot
        if binding is None:
            if slot is not None and slot.cleanup_error is not None:
                raise slot.cleanup_error
            if slot is not None and slot.receipt is not None:
                raise IrisCommandCleanupError("完成 Hook 的停止收据缺少命令依赖")
            if control.unknown_error is not None:
                logger.warning("run.finished 附加动作结果未知：%s", control.unknown_error)
            return

        async def drain() -> None:
            """使用同一停止操作，不因等待者取消重新停止环境。"""
            if slot is not None and slot.receipt is not None:
                await binding.service.wait_drained(slot.receipt)
                slot.receipt = None
                slot.cleanup_error = None
            else:
                await binding.service.stop(
                    CommandScope(entry.run_id, entry.session_id)
                ).wait_drained()

        task = asyncio.create_task(drain())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                entry.cancel_error = entry.cancel_error or error
        task.result()
        if control.unknown_error is not None:
            logger.warning("run.finished 附加动作结果未知，命令已收口：%s", control.unknown_error)

    def _record_failure(self, entry: FinishedHook, error: IrisCommandCleanupError) -> None:
        """保存真实资源错误，原 SDK 取消保留为异常链而不掩盖失败。"""
        entry.error = error
        if entry.cancel_error is not None:
            error.__cause__ = entry.cancel_error
        if self.admission_error is None:
            self.admission_error = error
        self.root._command_lifecycle.report_failure(entry.run_id, error)

    def _notify(self, session_id: str | None) -> None:
        """普通完成通知本 session；root 资源错误唤醒所有 session。"""
        groups = (
            tuple(self._subscribers.values())
            if session_id is None
            else (self._subscribers.get(session_id, set()),)
        )
        for callbacks in groups:
            for callback in tuple(callbacks):
                try:
                    callback()
                except Exception:
                    logger.exception("finished Hook 完成通知失败 session=%s", session_id)


def _retrieve_exception(task: asyncio.Task[None]) -> None:
    """即使没有驱动者等待，也取回实际 owner 的异常。"""
    if not task.cancelled():
        task.exception()
