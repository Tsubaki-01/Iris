"""Logical Agent run 的唯一 lifecycle owner。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from ..exceptions import (
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from ..lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    FinishRun,
    LifecycleStore,
    RunCheckpoint,
    RunErrorInfo,
    RunEvent,
    RunPhase,
    RunResult,
    RunSnapshot,
    RunStopReason,
    ToolCallPhase,
    snapshot_run,
)
from ..runtime import (
    AgentRuntime,
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeCursor,
)
from ..tools import CancellationRequestedError, CancellationSignal
from ._commit_port import StoreRuntimeCommitPort
from ._fingerprint import compute_environment_fingerprint
from .observer import RunEventObserver

logger = logging.getLogger(__name__)


class Clock(Protocol):
    """Runner 所需的 aware UTC wall-clock 边界。"""

    def now(self) -> datetime:
        """返回当前 aware 时间。"""


class _SystemClock:
    """默认 UTC wall clock。"""

    def now(self) -> datetime:
        return datetime.now(UTC)


class _MutableCancellationSignal(CancellationSignal):
    """一次 activation 共享的进程内 cooperative cancellation signal。"""

    def __init__(self) -> None:
        self._requested = False
        self._deadline_requested = False

    @property
    def requested(self) -> bool:
        return self._requested or self._deadline_requested

    @property
    def deadline_requested(self) -> bool:
        """返回该 signal 是否由 absolute deadline 触发。"""
        return self._deadline_requested

    def request(self) -> None:
        self._requested = True

    def request_deadline(self) -> None:
        """记录 deadline 原因，同时唤醒 cooperative cancellation 检查。"""
        self._deadline_requested = True

    def raise_if_requested(self) -> None:
        if self.requested:
            raise CancellationRequestedError("activation 已请求取消")


@dataclass(slots=True)
class ActiveActivation:
    """当前进程中 activation 的 live resources；不进入持久化。"""

    run_id: str
    activation_id: str
    signal: _MutableCancellationSignal
    task: asyncio.Task[RuntimeActivationResult] | None = None
    deadline_task: asyncio.Task[None] | None = None
    events: list[RunEvent] = field(default_factory=list)


class AgentRunner:
    """创建 logical run 并把一次 engine activation 结算为 durable result。"""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        store: LifecycleStore,
        observers: Sequence[RunEventObserver] = (),
        clock: Clock | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.observers = tuple(observers)
        self.clock = clock or _SystemClock()
        self.environment_fingerprint = compute_environment_fingerprint(runtime)
        self._active: dict[str, ActiveActivation] = {}

    async def start(
        self,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions | None = None,
    ) -> RunResult:
        """原子创建并推进一个 start activation 到 waiting 或 terminal。"""
        resolved_options = options or AgentRunOptions()
        run_id = request.run_id or f"run_{uuid.uuid4().hex}"
        resolved_request = request.model_copy(update={"run_id": run_id})
        activation_id = f"act_{uuid.uuid4().hex}"
        cursor = RuntimeCursor(position="before_model", step_index=0)
        session = self.store.load_session(resolved_request.session_id)
        checkpoint = RunCheckpoint(
            run_id=run_id,
            sequence=1,
            activation_id=activation_id,
            engine_cursor=cursor.model_dump(mode="json"),
            session_revision=session.revision,
            model_steps_reserved=0,
            model_steps_committed=0,
            environment_fingerprint=self.environment_fingerprint,
        )
        created = self.store.create_run(
            CreateRun(
                request=resolved_request,
                options=resolved_options,
                agent_id=self.runtime.environment.agent_config.name,
                environment_fingerprint=self.environment_fingerprint,
                start_activation_id=activation_id,
                initial_checkpoint=checkpoint,
                now=self._now(),
            )
        )
        if created.run.phase is RunPhase.TERMINAL:
            await self._deliver_events(list(created.events))
            return self._require_result(run_id)
        if created.checkpoint != checkpoint:
            raise IrisRunConflictError("create_run 返回了意外 initial checkpoint")

        active = ActiveActivation(
            run_id=run_id,
            activation_id=activation_id,
            signal=_MutableCancellationSignal(),
            events=list(created.events),
        )
        port = StoreRuntimeCommitPort(
            store=self.store,
            run=created.run,
            activation_id=activation_id,
            clock=self._now,
            event_sink=active.events,
        )
        activation = RuntimeActivationInput(
            run_id=run_id,
            activation_id=activation_id,
            session_id=resolved_request.session_id,
            kind="start",
            input=resolved_request.input,
            cursor=cursor,
            options=resolved_options.runtime,
        )
        self._register(active, created.run.current_activation_id)
        return await self._run_activation(active, activation=activation, port=port)

    def get_run(self, run_id: str) -> RunSnapshot:
        """读取一个 logical run 的 durable snapshot。"""
        record = self.store.load_run(self._required_id(run_id))
        if record is None:
            raise IrisRunNotFoundError("run 不存在", run_id=run_id)
        return snapshot_run(record)

    def get_result(self, run_id: str) -> RunResult | None:
        """读取 waiting/terminal durable result；active run 返回 ``None``。"""
        normalized = self._required_id(run_id)
        if self.store.load_run(normalized) is None:
            raise IrisRunNotFoundError("run 不存在", run_id=run_id)
        return self.store.load_result(normalized)

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        """读取 sequence 严格大于游标的 durable events。"""
        return self.store.list_events(self._required_id(run_id), after_sequence)

    async def _run_activation(
        self,
        active: ActiveActivation,
        *,
        activation: RuntimeActivationInput,
        port: StoreRuntimeCommitPort,
    ) -> RunResult:
        try:
            active.deadline_task = self._start_deadline_task(active.signal, port)
            active.task = asyncio.create_task(
                self.runtime.execute(
                    activation,
                    commits=port,
                    cancellation=active.signal,
                )
            )
            try:
                engine_result = await active.task
                self._settle_engine_result(active, engine_result, port)
            except (
                IrisRunConflictError,
                IrisRunNotFoundError,
                IrisRunPersistenceError,
                IrisRunStateError,
            ):
                raise
            except Exception as exc:
                self._finish_unexpected(active, exc, port)
        finally:
            port.revoke()
            try:
                await self._settle_live_resources(active)
            finally:
                current = self._active.get(active.run_id)
                if current is active:
                    self._active.pop(active.run_id, None)
            await self._deliver_events(active.events)
        return self._require_result(active.run_id)

    def _settle_engine_result(
        self,
        active: ActiveActivation,
        result: RuntimeActivationResult,
        port: StoreRuntimeCommitPort,
    ) -> None:
        current = self.store.load_run(active.run_id)
        if current is None:
            raise IrisRunNotFoundError("run 在 activation 期间消失", run_id=active.run_id)
        checkpoint = self.store.load_checkpoint(active.run_id)
        if (
            checkpoint is None
            or RuntimeCursor.model_validate(checkpoint.engine_cursor) != result.cursor
        ):
            raise IrisRunConflictError("engine outcome cursor 与 durable checkpoint 不匹配")
        if current.phase is RunPhase.TERMINAL:
            return
        if result.outcome is RuntimeActivationOutcome.SUSPENDED:
            if current.phase is not RunPhase.WAITING:
                raise IrisRunStateError("suspended engine outcome 缺少 durable waiting state")
            return
        if current.phase is not RunPhase.ACTIVE:
            raise IrisRunStateError("non-suspended engine outcome 遇到非 active run")
        outcome = result.outcome
        if (
            outcome is RuntimeActivationOutcome.CANCELLED
            and active.signal.deadline_requested
            and port.remaining_deadline_seconds() == 0
        ):
            outcome = RuntimeActivationOutcome.DEADLINE_EXCEEDED
        stop_reason = {
            RuntimeActivationOutcome.COMPLETED: RunStopReason.COMPLETED,
            RuntimeActivationOutcome.FAILED: RunStopReason.FAILED,
            RuntimeActivationOutcome.BUDGET_EXHAUSTED: RunStopReason.BUDGET_EXHAUSTED,
            RuntimeActivationOutcome.CANCELLED: RunStopReason.CANCELLED,
            RuntimeActivationOutcome.DEADLINE_EXCEEDED: RunStopReason.DEADLINE_EXCEEDED,
            RuntimeActivationOutcome.OUTCOME_UNKNOWN: RunStopReason.OUTCOME_UNKNOWN,
        }.get(outcome)
        if stop_reason is None:
            raise IrisRunStateError("未知 engine activation outcome")
        committed = self.store.finish_run(
            FinishRun(
                run_id=active.run_id,
                expected_run_revision=current.revision,
                activation_id=active.activation_id,
                stop_reason=stop_reason,
                assistant_message=result.assistant_message,
                error=result.error,
                now=self._now(),
            )
        )
        self._record_events(active.events, committed.events)

    def _finish_unexpected(
        self,
        active: ActiveActivation,
        error: Exception,
        port: StoreRuntimeCommitPort,
    ) -> None:
        del port
        current = self.store.load_run(active.run_id)
        if current is None or current.phase is not RunPhase.ACTIVE:
            raise error
        claimed = [
            record
            for record in self.store.list_tool_calls(active.run_id)
            if record.phase is ToolCallPhase.CLAIMED
            and record.claim_activation_id == active.activation_id
        ]
        if claimed:
            stop_reason = RunStopReason.OUTCOME_UNKNOWN
            run_error = RunErrorInfo(
                code="TOOL_OUTCOME_UNKNOWN",
                message="工具 claim 后发生意外异常，effect 结果不可证明",
                source="tool",
                details={
                    "tool_call_ids": [record.tool_call_id for record in claimed],
                    "cause": str(error) or type(error).__name__,
                },
            )
        else:
            stop_reason = RunStopReason.FAILED
            run_error = RunErrorInfo(
                code="RUNTIME_ERROR",
                message=str(error) or type(error).__name__,
                source="runtime",
            )
        committed = self.store.finish_run(
            FinishRun(
                run_id=active.run_id,
                expected_run_revision=current.revision,
                activation_id=active.activation_id,
                stop_reason=stop_reason,
                error=run_error,
                now=self._now(),
            )
        )
        self._record_events(active.events, committed.events)

    def _start_deadline_task(
        self,
        signal: _MutableCancellationSignal,
        port: StoreRuntimeCommitPort,
    ) -> asyncio.Task[None] | None:
        remaining = port.remaining_deadline_seconds()
        if remaining is None:
            return None

        async def request_at_deadline() -> None:
            await asyncio.sleep(remaining)
            signal.request_deadline()

        return asyncio.create_task(request_at_deadline())

    async def _settle_live_resources(self, active: ActiveActivation) -> None:
        if active.deadline_task is not None:
            active.deadline_task.cancel()
            with suppress(asyncio.CancelledError):
                await active.deadline_task
            active.deadline_task = None
        active.task = None

    async def _deliver_events(self, events: list[RunEvent]) -> None:
        seen: set[tuple[str, int]] = set()
        for event in sorted(events, key=lambda item: item.sequence):
            key = (event.run_id, event.sequence)
            if key in seen:
                continue
            seen.add(key)
            for observer in self.observers:
                try:
                    await observer.on_event(event)
                except Exception:
                    logger.exception(
                        "run event observer 处理失败",
                        extra={"run_id": event.run_id, "sequence": event.sequence},
                    )

    def _register(self, active: ActiveActivation, current_activation_id: str | None) -> None:
        if current_activation_id != active.activation_id:
            raise IrisRunConflictError("active map registration fence 不匹配")
        if active.run_id in self._active:
            raise IrisRunConflictError("run 已存在 process-local active activation")
        self._active[active.run_id] = active

    def _require_result(self, run_id: str) -> RunResult:
        result = self.store.load_result(run_id)
        if result is None:
            raise IrisRunStateError("activation settlement 后缺少 durable result", run_id=run_id)
        return result

    def _now(self) -> datetime:
        now = self.clock.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise IrisRunStateError("runner clock 必须返回 aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _required_id(run_id: str) -> str:
        normalized = run_id.strip()
        if not normalized:
            raise IrisRunStateError("run_id 不能为空")
        return normalized

    @staticmethod
    def _record_events(target: list[RunEvent], events: tuple[RunEvent, ...]) -> None:
        keys = {(event.run_id, event.sequence) for event in target}
        target.extend(event for event in events if (event.run_id, event.sequence) not in keys)


__all__ = ["AgentRunner", "Clock"]
