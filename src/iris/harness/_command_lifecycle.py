"""root 进程内的命令清理结算与等待期限，不持久化资源句柄。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..command import CommandStopReceipt
from ..exceptions import IrisCommandCleanupError
from ..lifecycle import RunErrorInfo, RunRecord, RunResult, RunStopReason, TokenUsage
from ..message import Msg
from ..tools.subagent import SubagentRoute
from ._events import _RunEventCollector

if TYPE_CHECKING:
    from ._subagent import HarnessSubagentController
    from .runner import AgentRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RootCommandTarget:
    """root 自身一直存在，不因一次 activation 结束而重建。"""

    runner: AgentRunner

    @asynccontextmanager
    async def open(self, run_id: str) -> AsyncIterator[AgentRunner]:
        """借用 root runner。"""
        yield self.runner


@dataclass(frozen=True, slots=True)
class ChildCommandTarget:
    """只保存可重建 child 的路由，不保存已关闭 runner 回调。"""

    controller: HarnessSubagentController
    route: SubagentRoute

    @asynccontextmanager
    async def open(self, run_id: str) -> AsyncIterator[AgentRunner]:
        """借用 exact live child 或为该 run 重建临时 runner。"""
        async with self.controller.open_command_target(self.route, run_id) as runner:
            yield runner


type CommandTarget = RootCommandTarget | ChildCommandTarget


@dataclass(slots=True)
class PendingSettlement:
    """一次未完成结算的原意图；重试不再调用 engine。"""

    target: CommandTarget
    run_id: str
    activation_id: str | None
    stop_reason: RunStopReason | None
    error: RunErrorInfo | None
    assistant_message: Msg | None
    interaction_close_reason: str | None
    events: _RunEventCollector
    model_failure_usage: TokenUsage | None = None
    receipt: CommandStopReceipt | None = None
    call_id: str | None = None
    initial_error: IrisCommandCleanupError | None = None
    attempt: asyncio.Task[RunResult | None] | None = None


@dataclass(slots=True)
class _Deadline:
    """区分尚未触发的睡眠和已经开始的结算。"""

    task: asyncio.Task[None] | None = None
    fired: bool = False


class CommandLifecycle:
    """同一 root 共享 pending、期限和仅供当前 child 交接的收据。"""

    def __init__(self, root: AgentRunner) -> None:
        self.root = root
        self.pending: dict[str, PendingSettlement] = {}
        self.deadlines: dict[str, _Deadline] = {}
        self.settled_receipts: dict[str, CommandStopReceipt] = {}

    async def join(self, pending: PendingSettlement) -> RunResult | None:
        """所有等待者加入一次完整结算；取消等待者不取消结算。"""
        if pending.attempt is None or (
            pending.attempt.done() and pending.attempt.exception() is not None
        ):
            pending.attempt = asyncio.create_task(self._attempt(pending))
            pending.attempt.add_done_callback(_retrieve_exception)
        return await asyncio.shield(pending.attempt)

    async def _attempt(self, pending: PendingSettlement) -> RunResult | None:
        """一次失败只报告一次；保留原意图供下一次显式重试。"""
        try:
            async with pending.target.open(pending.run_id) as runner:
                with runner._observe_command_attempt(pending):
                    if pending.initial_error is not None:
                        error = pending.initial_error
                        pending.initial_error = None
                        raise error
                    return await runner._perform_command_settlement(pending)
        except IrisCommandCleanupError as exc:
            # 后终态 Hook 已移除 pending，其资源错误由共享 Hook owner 报告。
            if pending.run_id in self.pending:
                self.report_failure(pending.run_id, exc)
            raise
        except Exception:
            # store/fence 等错误继续走原 durable recovery，不由 close 隐式重试提交。
            self.pending.pop(pending.run_id, None)
            raise

    def report_failure(self, run_id: str, error: IrisCommandCleanupError) -> None:
        """child 与后台清理经 root 通知 attachment，并按原路径发布诊断。"""
        from .streaming import CommandCleanupFailed

        run = self.root.store.load_run(run_id)
        if run is not None:
            fact = CommandCleanupFailed(
                run_id=run_id,
                session_id=run.session_id,
                error=RunErrorInfo(code=error.runtime_code, message=str(error), source="tool"),
            )
            self.root._notify_session_fact(fact)
            self.root._publish_live_fact(fact)
        if run is None or self.root._live_publisher is None:
            logger.error("命令环境清理失败 run=%s: %s", run_id, error)

    def register_deadline(self, run: RunRecord, target: CommandTarget) -> None:
        """一个 run 只注册一个 timer，WAITING 不撤销。"""
        deadline = run.options.limits.deadline_at
        if deadline is None or run.run_id in self.deadlines:
            return
        entry = _Deadline()
        self.deadlines[run.run_id] = entry

        async def fire() -> None:
            while (remaining := (deadline - self.root._now()).total_seconds()) > 0:
                await asyncio.sleep(remaining)
            entry.fired = True
            try:
                pending = self.pending.get(run.run_id)
                if pending is not None:
                    if pending.attempt is not None and not pending.attempt.done():
                        await asyncio.shield(pending.attempt)
                    return
                async with target.open(run.run_id) as runner:
                    await runner._command_deadline_due(run.run_id)
            except IrisCommandCleanupError:
                # attempt 已报告；timer 只负责取回异常。
                pass
            except Exception:
                logger.exception("run 期限结算失败 run=%s", run.run_id)

        with self.root.runtime.environment.observability.detached():
            entry.task = asyncio.create_task(fire())

    def terminal(self, run_id: str) -> None:
        """终态撤销未触发 timer，当前结算任务不取消自己。"""
        entry = self.deadlines.pop(run_id, None)
        if entry is not None and not entry.fired and entry.task is not None:
            entry.task.cancel()

    def take_settlement_receipt(self, run_id: str) -> CommandStopReceipt | None:
        """仅把本次 child 结算的停止事实交给其父调用一次。"""
        return self.settled_receipts.pop(run_id, None)

    async def aclose(self) -> None:
        """撤销睡眠，等待已触发任务，并显式重试剩余 pending。"""
        tasks = []
        for entry in tuple(self.deadlines.values()):
            if entry.task is not None:
                if not entry.fired:
                    entry.task.cancel()
                tasks.append(entry.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.deadlines.clear()
        for pending in tuple(self.pending.values()):
            await self.join(pending)


def _retrieve_exception(task: asyncio.Task[RunResult | None]) -> None:
    """即使所有等待者都取消，也取回 attempt 的最终异常。"""
    if not task.cancelled():
        task.exception()
