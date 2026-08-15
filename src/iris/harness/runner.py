"""Logical Agent run 的唯一 lifecycle owner。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ..agents import AgentConfig, load_agent_config
from ..exceptions import (
    HITLConflictError,
    IrisCancellationRequestedError,
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunObservationTimeoutError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from ..hitl import (
    ApprovedToolCall,
    HumanInteraction,
    HumanInteractionResponse,
    HumanInteractionService,
    InteractionStatus,
)
from ..lifecycle import (
    ActivationKind,
    AgentRunOptions,
    AgentRunRequest,
    CheckpointResumability,
    CreateRun,
    FinishRun,
    LifecycleStore,
    RecoverActiveRun,
    RecoveryDisposition,
    RequestCancellation,
    ResolveInteraction,
    ResumeWaitingRun,
    RunCheckpoint,
    RunErrorInfo,
    RunEvent,
    RunPhase,
    RunRecord,
    RunResult,
    RunSnapshot,
    RunStopReason,
    ToolCallPhase,
    snapshot_run,
)
from ..memory import MemoryService
from ..runtime import (
    AgentRuntime,
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeCursor,
    RuntimeFactory,
    RuntimeProvider,
    RuntimeSteeringPort,
)
from ..store import InMemoryLifecycleStore, SQLiteStore
from ..tools import CancellationSignal
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
            raise IrisCancellationRequestedError("activation 已请求取消")


@dataclass(slots=True)
class ActiveActivation:
    """当前进程中 activation 的 live resources；不进入持久化。"""

    run_id: str
    activation_id: str
    signal: _MutableCancellationSignal
    task: asyncio.Task[RuntimeActivationResult] | None = None
    deadline_task: asyncio.Task[None] | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    events: list[RunEvent] = field(default_factory=list)
    steering: RuntimeSteeringPort | None = None
    durable_event_callback: Callable[[RunEvent], None] | None = None


class AgentRunner:
    """创建 logical run 并把一次 engine activation 结算为 durable result。"""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        store: LifecycleStore,
        observers: Sequence[RunEventObserver] = (),
        clock: Clock | None = None,
        interaction_service: HumanInteractionService | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.observers = tuple(observers)
        self.clock = clock or _SystemClock()
        self.interaction_service = interaction_service or HumanInteractionService()
        if hasattr(self.interaction_service, "store") or any(
            not callable(getattr(self.interaction_service, name, None))
            for name in ("create_pending", "validate_response", "project_response")
        ):
            raise IrisRunStateError("runner interaction service 必须是无状态领域服务")
        self.environment_fingerprint = compute_environment_fingerprint(runtime)
        self._active: dict[str, ActiveActivation] = {}

    @classmethod
    def from_config_path(
        cls,
        path: str | Path,
        *,
        provider: RuntimeProvider | None = None,
        memory_service: MemoryService | None = None,
        store: LifecycleStore | None = None,
        observers: Sequence[RunEventObserver] = (),
        clock: Clock | None = None,
        api_key: str | None = None,
    ) -> AgentRunner:
        """从 agent 配置路径装配 engine 与唯一 lifecycle store。"""
        config_path = Path(path)
        return cls.from_config(
            load_agent_config(config_path),
            config_path=config_path,
            provider=provider,
            memory_service=memory_service,
            store=store,
            observers=observers,
            clock=clock,
            api_key=api_key,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        config_path: Path | None = None,
        provider: RuntimeProvider | None = None,
        memory_service: MemoryService | None = None,
        store: LifecycleStore | None = None,
        observers: Sequence[RunEventObserver] = (),
        clock: Clock | None = None,
        api_key: str | None = None,
    ) -> AgentRunner:
        """从已校验配置装配 engine；durable ownership 只属于 harness。"""
        runtime = RuntimeFactory.from_config(
            config,
            config_path=config_path,
            provider=provider,
            memory_service=memory_service,
            api_key=api_key,
        )
        resolved_store = (
            store if store is not None else _build_lifecycle_store(config, config_path=config_path)
        )
        return cls(
            runtime=runtime,
            store=resolved_store,
            observers=observers,
            clock=clock,
        )

    async def start(
        self,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions | None = None,
    ) -> RunResult:
        """原子创建并推进一个 start activation 到 waiting 或 terminal。"""
        return await self._start_managed(request, options=options)

    async def _start_managed(
        self,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions | None = None,
        steering: RuntimeSteeringPort | None = None,
        durable_event_callback: Callable[[RunEvent], None] | None = None,
        activation_started: asyncio.Event | None = None,
    ) -> RunResult:
        """创建 start activation，并注入可选的 process-local managed hooks。"""
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
        events: list[RunEvent] = []
        self._record_events(events, created.events, durable_event_callback)
        if created.run.phase is RunPhase.TERMINAL:
            await self._deliver_events(events)
            return self._require_result(run_id)
        if created.checkpoint != checkpoint:
            raise IrisRunConflictError("create_run 返回了意外 initial checkpoint")

        active = ActiveActivation(
            run_id=run_id,
            activation_id=activation_id,
            signal=_MutableCancellationSignal(),
            events=events,
            steering=steering,
            durable_event_callback=durable_event_callback,
        )
        port = StoreRuntimeCommitPort(
            store=self.store,
            run=created.run,
            activation_id=activation_id,
            clock=self._now,
            event_sink=active.events,
            durable_event_callback=durable_event_callback,
            interaction_service=self.interaction_service,
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
        if activation_started is not None:
            activation_started.set()
        return await self._run_activation(active, activation=activation, port=port)

    async def resume(
        self,
        run_id: str,
        *,
        interaction_id: str,
        response: HumanInteractionResponse,
    ) -> RunResult:
        """从 durable waiting checkpoint 创建一个新的 resume activation。"""
        return await self._resume_managed(
            run_id,
            interaction_id=interaction_id,
            response=response,
        )

    async def _resume_managed(
        self,
        run_id: str,
        *,
        interaction_id: str,
        response: HumanInteractionResponse,
        steering: RuntimeSteeringPort | None = None,
        durable_event_callback: Callable[[RunEvent], None] | None = None,
        activation_started: asyncio.Event | None = None,
    ) -> RunResult:
        """创建 resume activation，并注入可选的 process-local managed hooks。"""
        normalized_run_id = self._required_id(run_id)
        normalized_interaction_id = self._required_id(interaction_id)
        run = self.store.load_run(normalized_run_id)
        if run is None:
            raise IrisRunNotFoundError("run 不存在", run_id=normalized_run_id)
        interaction = self.store.load_interaction(normalized_interaction_id)
        if interaction is None:
            raise IrisRunNotFoundError(
                "interaction 不存在",
                interaction_id=normalized_interaction_id,
            )
        if interaction.run_id != normalized_run_id:
            raise IrisRunConflictError("interaction 不属于目标 run")
        if interaction.status is InteractionStatus.CLOSED:
            return self._closed_retry_result(run, interaction, response)
        if run.phase is RunPhase.TERMINAL:
            raise IrisRunStateError("terminal run 不接受 resume", run_id=run.run_id)
        if run.phase is RunPhase.ACTIVE:
            raise IrisRunStateError("active run 必须通过 recover 处理", run_id=run.run_id)
        if run.pending_interaction_id != interaction.interaction_id:
            raise IrisRunConflictError("waiting run 的 interaction identity 已变化")

        event_cursor = run.last_event_sequence
        now = self._now()
        settled = self._settle_waiting_if_due(run, interaction, now=now)
        if settled is not None:
            events: list[RunEvent] = []
            self._record_events(
                events,
                self.store.list_events(run.run_id, event_cursor),
                durable_event_callback,
            )
            await self._deliver_events(events)
            return settled

        checkpoint = self.store.load_checkpoint(run.run_id)
        if checkpoint is None:
            raise IrisRunRecoveryError(
                "waiting run 缺少 durable checkpoint",
                run_id=run.run_id,
            )
        cursor = self._validate_resume_checkpoint(run, interaction, checkpoint)
        self.interaction_service.validate_response(
            interaction,
            run=snapshot_run(run),
            response=response,
            now=now,
            environment_fingerprint=self.environment_fingerprint,
        )
        if interaction.status is InteractionStatus.PENDING:
            resolved = self.store.resolve_interaction(
                ResolveInteraction(
                    run_id=run.run_id,
                    expected_run_revision=run.revision,
                    interaction_id=interaction.interaction_id,
                    expected_interaction_version=interaction.version,
                    response=response,
                    expected_fingerprint=interaction.request.tool_call.fingerprint,
                    now=now,
                )
            )
            run = resolved.run
            events = []
            self._record_events(events, resolved.events, durable_event_callback)
            if resolved.interaction is None:
                raise IrisRunStateError("resolve commit 缺少 interaction")
            interaction = resolved.interaction
        else:
            events = []
        projection = self.interaction_service.project_response(interaction, response)
        activation_id = f"act_{uuid.uuid4().hex}"
        begun = self.store.resume_waiting_run(
            ResumeWaitingRun(
                run_id=run.run_id,
                expected_run_revision=run.revision,
                new_activation_id=activation_id,
                kind=ActivationKind.RESUME,
                expected_checkpoint_sequence=checkpoint.sequence,
                now=self._now(),
            )
        )
        self._record_events(events, begun.events, durable_event_callback)
        if begun.checkpoint is None:
            raise IrisRunStateError("begin activation 缺少 rebound checkpoint")
        if begun.checkpoint.engine_cursor != checkpoint.engine_cursor:
            raise IrisRunConflictError("rebound checkpoint cursor 与 waiting checkpoint 不匹配")
        runtime_projection = (
            RuntimeApprovedToolCall.model_validate(projection.model_dump())
            if isinstance(projection, ApprovedToolCall)
            else projection
        )
        active = ActiveActivation(
            run_id=run.run_id,
            activation_id=activation_id,
            signal=_MutableCancellationSignal(),
            events=events,
            steering=steering,
            durable_event_callback=durable_event_callback,
        )
        port = StoreRuntimeCommitPort(
            store=self.store,
            run=begun.run,
            activation_id=activation_id,
            clock=self._now,
            event_sink=active.events,
            durable_event_callback=durable_event_callback,
            interaction_service=self.interaction_service,
        )
        activation = RuntimeActivationInput(
            run_id=run.run_id,
            activation_id=activation_id,
            session_id=run.session_id,
            kind="resume",
            input=None,
            cursor=cursor,
            options=run.options.runtime,
            interaction_projection=runtime_projection,
        )
        self._register(active, begun.run.current_activation_id)
        if activation_started is not None:
            activation_started.set()
        return await self._run_activation(active, activation=activation, port=port)

    def request_cancel(
        self,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> RunSnapshot:
        """持久化首次 cancellation request，再 signal 当前进程 activation。"""
        normalized = self._required_id(run_id)
        normalized_reason = "cancel requested" if reason is None else reason.strip()
        if not normalized_reason:
            raise IrisRunStateError("cancellation reason 不能为空")
        run = self.store.load_run(normalized)
        if run is None:
            raise IrisRunNotFoundError("run 不存在", run_id=normalized)
        if run.phase is RunPhase.TERMINAL:
            return snapshot_run(run)
        if run.cancellation_requested_at is None:
            committed = self.store.request_cancellation(
                RequestCancellation(
                    run_id=run.run_id,
                    expected_run_revision=run.revision,
                    activation_id=(
                        run.current_activation_id if run.phase is RunPhase.ACTIVE else None
                    ),
                    reason=normalized_reason,
                    settle_waiting=run.phase is RunPhase.WAITING,
                    now=self._now(),
                )
            )
            run = committed.run
            active = self._active.get(run.run_id)
            if active is not None:
                self._record_events(
                    active.events,
                    committed.events,
                    active.durable_event_callback,
                )
        active = self._active.get(run.run_id)
        if (
            active is not None
            and run.phase is RunPhase.ACTIVE
            and run.current_activation_id == active.activation_id
        ):
            self._interrupt_active(active)
        return snapshot_run(run)

    async def cancel(
        self,
        run_id: str,
        *,
        reason: str | None = None,
        settlement_timeout: float | None = None,
    ) -> RunResult:
        """请求取消并只观察 durable settlement；超时不写入新事实。"""
        if settlement_timeout is not None and settlement_timeout <= 0:
            raise IrisRunStateError("settlement_timeout 必须大于 0")
        normalized = self._required_id(run_id)
        before = self.store.load_run(normalized)
        if before is None:
            raise IrisRunNotFoundError("run 不存在", run_id=normalized)
        snapshot = self.request_cancel(normalized, reason=reason)
        if snapshot.phase is RunPhase.TERMINAL:
            await self._deliver_events(
                self.store.list_events(normalized, before.last_event_sequence)
            )
            return self._require_result(normalized)
        return await self._observe_settlement(
            normalized,
            settlement_timeout=settlement_timeout,
        )

    async def recover(
        self,
        run_id: str,
        *,
        expected_activation_id: str | None = None,
    ) -> RunResult:
        """根据精确 activation fence 与 durable facts 显式恢复 run。"""
        normalized = self._required_id(run_id)
        run = self.store.load_run(normalized)
        if run is None:
            raise IrisRunNotFoundError("run 不存在", run_id=normalized)
        if run.phase is RunPhase.TERMINAL:
            return self._require_result(normalized)
        if run.phase is RunPhase.WAITING:
            interaction = self.store.load_interaction(run.pending_interaction_id or "")
            if interaction is None:
                raise IrisRunRecoveryError(
                    "waiting run 缺少 durable interaction", run_id=run.run_id
                )
            cursor = run.last_event_sequence
            settled = self._settle_waiting_if_due(run, interaction, now=self._now())
            if settled is not None:
                await self._deliver_events(self.store.list_events(run.run_id, cursor))
                return settled
            raise IrisRunStateError(
                "waiting run 必须通过 resume 继续",
                run_id=run.run_id,
            )
        if expected_activation_id is None or not expected_activation_id.strip():
            raise IrisRunConflictError("active recovery 必须提供 expected_activation_id")
        expected = expected_activation_id.strip()
        if run.current_activation_id != expected:
            raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
        if normalized in self._active:
            raise IrisRunStateError("当前进程仍拥有 live activation，不能 takeover")
        checkpoint = self.store.load_checkpoint(run.run_id)
        if checkpoint is None:
            raise IrisRunRecoveryError("active run 缺少 durable checkpoint", run_id=run.run_id)
        claimed = [
            record
            for record in self.store.list_tool_calls(run.run_id)
            if record.phase is ToolCallPhase.CLAIMED
        ]
        if claimed:
            disposition = RecoveryDisposition.OUTCOME_UNKNOWN
            recovered_cursor = None
        else:
            recovered_cursor = self._validate_recovery_checkpoint(run, checkpoint)
            if (
                checkpoint.resumability is CheckpointResumability.OUTCOME_READY
                and recovered_cursor.position == "outcome_ready"
            ):
                disposition = RecoveryDisposition.FINALIZE
            elif (
                checkpoint.resumability is CheckpointResumability.SAFE
                and recovered_cursor.position != "outcome_ready"
            ):
                disposition = RecoveryDisposition.RESUME
            else:
                raise IrisRunRecoveryError(
                    "checkpoint resumability 与 cursor position 不匹配",
                    run_id=run.run_id,
                )
        new_activation_id = (
            f"act_{uuid.uuid4().hex}" if disposition is RecoveryDisposition.RESUME else None
        )
        recovered = self.store.recover_active_run(
            RecoverActiveRun(
                run_id=run.run_id,
                expected_run_revision=run.revision,
                expected_activation_id=expected,
                expected_checkpoint_sequence=checkpoint.sequence,
                recovery_disposition=disposition,
                new_activation_id=new_activation_id,
                now=self._now(),
            )
        )
        if recovered.run.phase is RunPhase.TERMINAL:
            await self._deliver_events(list(recovered.events))
            return self._require_result(run.run_id)
        if recovered.checkpoint is None or recovered_cursor is None or new_activation_id is None:
            raise IrisRunRecoveryError("recover commit 缺少 rebound activation facts")
        if RuntimeCursor.model_validate(recovered.checkpoint.engine_cursor) != recovered_cursor:
            raise IrisRunConflictError("recover rebound checkpoint cursor 已变化")
        active = ActiveActivation(
            run_id=run.run_id,
            activation_id=new_activation_id,
            signal=_MutableCancellationSignal(),
            events=list(recovered.events),
        )
        port = StoreRuntimeCommitPort(
            store=self.store,
            run=recovered.run,
            activation_id=new_activation_id,
            clock=self._now,
            event_sink=active.events,
            interaction_service=self.interaction_service,
        )
        activation = RuntimeActivationInput(
            run_id=run.run_id,
            activation_id=new_activation_id,
            session_id=run.session_id,
            kind="recover",
            input=(
                run.request.input
                if recovered_cursor.position == "before_model" and recovered_cursor.step_index == 0
                else None
            ),
            cursor=recovered_cursor,
            options=run.options.runtime,
        )
        self._register(active, recovered.run.current_activation_id)
        return await self._run_activation(active, activation=activation, port=port)

    def _validate_recovery_checkpoint(
        self,
        run: RunRecord,
        checkpoint: RunCheckpoint,
    ) -> RuntimeCursor:
        """验证 safe/outcome-ready recovery 的交叉 durable facts。"""
        session = self.store.load_session(run.session_id)
        if (
            checkpoint.run_id != run.run_id
            or checkpoint.sequence != run.checkpoint_sequence
            or checkpoint.activation_id != run.current_activation_id
            or checkpoint.session_revision != session.revision
            or checkpoint.model_steps_reserved != run.usage.model_steps_reserved
            or checkpoint.model_steps_committed != run.usage.model_steps_committed
            or checkpoint.environment_fingerprint != run.environment_fingerprint
            or checkpoint.environment_fingerprint != self.environment_fingerprint
        ):
            raise IrisRunRecoveryError(
                "active checkpoint 与 durable run/session facts 不匹配",
                run_id=run.run_id,
            )
        try:
            return RuntimeCursor.model_validate(checkpoint.engine_cursor)
        except (TypeError, ValueError) as exc:
            raise IrisRunRecoveryError(
                "active checkpoint cursor 无法恢复",
                run_id=run.run_id,
            ) from exc

    def _interrupt_active(
        self,
        active: ActiveActivation,
        *,
        deadline: bool = False,
    ) -> None:
        """先标记 signal；仅在尚无 durable claim 时中断 async operation。"""
        if deadline:
            active.signal.request_deadline()
        else:
            active.signal.request()
        claimed = any(
            record.phase is ToolCallPhase.CLAIMED
            and record.claim_activation_id == active.activation_id
            for record in self.store.list_tool_calls(active.run_id)
        )
        task = active.task
        if claimed or task is None or task.done():
            return
        task.get_loop().call_soon_threadsafe(task.cancel)

    async def _observe_settlement(
        self,
        run_id: str,
        *,
        settlement_timeout: float | None,
    ) -> RunResult:
        """优先等待本地 settlement，并以纯读取覆盖跨进程 run。"""
        loop = asyncio.get_running_loop()
        deadline = None if settlement_timeout is None else loop.time() + settlement_timeout
        while True:
            result = self.store.load_result(run_id)
            if result is not None and result.run.phase is RunPhase.TERMINAL:
                return result
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise IrisRunObservationTimeoutError(
                    "等待 run cancellation settlement 超时",
                    run_id=run_id,
                )
            interval = 0.05 if remaining is None else min(0.05, remaining)
            active = self._active.get(run_id)
            if active is None:
                await asyncio.sleep(interval)
                continue
            try:
                await asyncio.wait_for(active.settled.wait(), timeout=interval)
            except TimeoutError:
                pass

    def _validate_resume_checkpoint(
        self,
        run: RunRecord,
        interaction: HumanInteraction,
        checkpoint: RunCheckpoint,
    ) -> RuntimeCursor:
        """在消费人工响应前验证 waiting checkpoint 的交叉事实。"""
        session = self.store.load_session(run.session_id)
        if (
            checkpoint.run_id != run.run_id
            or checkpoint.sequence != run.checkpoint_sequence
            or checkpoint.session_revision != session.revision
            or checkpoint.model_steps_reserved != run.usage.model_steps_reserved
            or checkpoint.model_steps_committed != run.usage.model_steps_committed
            or checkpoint.environment_fingerprint != run.environment_fingerprint
            or checkpoint.environment_fingerprint != self.environment_fingerprint
            or checkpoint.resumability is not CheckpointResumability.SAFE
        ):
            raise IrisRunRecoveryError(
                "waiting checkpoint 与 durable run/session facts 不匹配",
                run_id=run.run_id,
            )
        try:
            cursor = RuntimeCursor.model_validate(checkpoint.engine_cursor)
        except (TypeError, ValueError) as exc:
            raise IrisRunRecoveryError(
                "waiting checkpoint cursor 无法恢复",
                run_id=run.run_id,
            ) from exc
        if cursor.position != "tool_batch" or cursor.next_tool_index >= len(cursor.tool_calls):
            raise IrisRunRecoveryError(
                "waiting checkpoint cursor 不在可恢复 interaction 位置",
                run_id=run.run_id,
            )
        current_call = cursor.tool_calls[cursor.next_tool_index]
        subject = interaction.request.tool_call
        current_record = next(
            (
                item
                for item in self.store.list_tool_calls(run.run_id)
                if item.tool_call_id == subject.tool_call_id
            ),
            None,
        )
        if (
            cursor.step_index != interaction.step_index
            or current_call.id != subject.tool_call_id
            or current_call.name != subject.tool_name
            or current_record is None
            or current_record.step_index != interaction.step_index
            or current_record.tool_name != subject.tool_name
            or current_record.arguments != subject.arguments
            or current_record.fingerprint != subject.fingerprint
            or current_record.interaction_id != interaction.interaction_id
            or current_record.phase is not ToolCallPhase.PREPARED
        ):
            raise IrisRunRecoveryError(
                "waiting checkpoint cursor 与 interaction subject 不匹配",
                run_id=run.run_id,
            )
        return cursor

    def _closed_retry_result(
        self,
        run: RunRecord,
        interaction: HumanInteraction,
        response: HumanInteractionResponse,
    ) -> RunResult:
        if interaction.response is not None and interaction.response != response:
            raise HITLConflictError("interaction 已由不同 response 解决")
        result = self.store.load_result(run.run_id)
        if result is None:
            raise IrisRunStateError(
                "response 已提交但 run 仍 active；需要 recover",
                run_id=run.run_id,
            )
        return result

    def _settle_waiting_if_due(
        self,
        run: RunRecord,
        interaction: HumanInteraction,
        *,
        now: datetime,
    ) -> RunResult | None:
        stop_reason: RunStopReason | None = None
        close_reason: str | None = None
        if run.cancellation_requested_at is not None:
            stop_reason = RunStopReason.CANCELLED
            close_reason = "cancelled"
        else:
            deadline = run.options.limits.deadline_at
            expiry = (
                interaction.expires_at if interaction.status is InteractionStatus.PENDING else None
            )
            deadline_due = deadline is not None and now >= deadline
            expiry_due = expiry is not None and now >= expiry
            if deadline_due and expiry_due:
                if deadline is None or expiry is None:  # pragma: no cover - 已由 due facts 收窄
                    raise IrisRunStateError("waiting due facts 不完整")
                stop_reason = (
                    RunStopReason.DEADLINE_EXCEEDED
                    if deadline <= expiry
                    else RunStopReason.INTERACTION_EXPIRED
                )
                close_reason = stop_reason.value
            elif deadline_due:
                stop_reason = RunStopReason.DEADLINE_EXCEEDED
                close_reason = "deadline_exceeded"
            elif expiry_due:
                stop_reason = RunStopReason.INTERACTION_EXPIRED
                close_reason = "interaction_expired"
        if stop_reason is None:
            return None
        committed = self.store.finish_run(
            FinishRun(
                run_id=run.run_id,
                expected_run_revision=run.revision,
                stop_reason=stop_reason,
                interaction_close_reason=close_reason,
                now=now,
            )
        )
        if committed.result is None:
            raise IrisRunStateError("waiting settlement 缺少 durable result", run_id=run.run_id)
        return committed.result

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
            active.deadline_task = self._start_deadline_task(active, port)
            active.task = asyncio.create_task(
                self.runtime.execute(
                    activation,
                    commits=port,
                    cancellation=active.signal,
                    steering=active.steering,
                )
            )
            try:
                engine_result = await active.task
                self._settle_engine_result(active, engine_result, port)
            except asyncio.CancelledError:
                if not active.signal.requested:
                    raise
                self._finish_cancelled_task(active, port)
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
                active.settled.set()
            await self._deliver_events(active.events)
        return self._require_result(active.run_id)

    def _finish_cancelled_task(
        self,
        active: ActiveActivation,
        port: StoreRuntimeCommitPort,
    ) -> None:
        """把被中断的 async operation 映射为可证明的 durable outcome。"""
        del port
        current = self.store.load_run(active.run_id)
        if current is None:
            raise IrisRunNotFoundError("run 在 cancellation 期间消失", run_id=active.run_id)
        if current.phase is RunPhase.TERMINAL:
            return
        claimed = [
            record
            for record in self.store.list_tool_calls(active.run_id)
            if record.phase is ToolCallPhase.CLAIMED
            and record.claim_activation_id == active.activation_id
        ]
        if claimed:
            stop_reason = RunStopReason.OUTCOME_UNKNOWN
            error = RunErrorInfo(
                code="TOOL_OUTCOME_UNKNOWN",
                message="工具 claim 后 activation 被中断，effect 结果不可证明",
                source="tool",
                details={"tool_call_ids": [record.tool_call_id for record in claimed]},
            )
        elif active.signal.deadline_requested:
            stop_reason = RunStopReason.DEADLINE_EXCEEDED
            error = None
        else:
            stop_reason = RunStopReason.CANCELLED
            error = None
        committed = self.store.finish_run(
            FinishRun(
                run_id=active.run_id,
                expected_run_revision=current.revision,
                activation_id=active.activation_id,
                stop_reason=stop_reason,
                error=error,
                now=self._now(),
            )
        )
        self._record_events(active.events, committed.events, active.durable_event_callback)

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
        if outcome is RuntimeActivationOutcome.CANCELLED and active.signal.deadline_requested:
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
        self._record_events(active.events, committed.events, active.durable_event_callback)

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
        self._record_events(active.events, committed.events, active.durable_event_callback)

    def _start_deadline_task(
        self,
        active: ActiveActivation,
        port: StoreRuntimeCommitPort,
    ) -> asyncio.Task[None] | None:
        remaining = port.remaining_deadline_seconds()
        if remaining is None:
            return None

        async def request_at_deadline() -> None:
            await asyncio.sleep(remaining)
            self._interrupt_active(active, deadline=True)

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
    def _record_events(
        target: list[RunEvent],
        events: Sequence[RunEvent],
        durable_event_callback: Callable[[RunEvent], None] | None = None,
    ) -> None:
        """去重收集 durable events，并同步隔离可选 callback。"""
        keys = {(event.run_id, event.sequence) for event in target}
        for event in events:
            key = (event.run_id, event.sequence)
            if key in keys:
                continue
            keys.add(key)
            target.append(event)
            if durable_event_callback is None:
                continue
            try:
                durable_event_callback(event)
            except Exception:
                logger.exception(
                    "durable event callback 处理失败",
                    extra={"run_id": event.run_id, "sequence": event.sequence},
                )


def _build_lifecycle_store(
    config: AgentConfig,
    *,
    config_path: Path | None,
) -> LifecycleStore:
    """按 session 配置选择 harness-owned lifecycle store。"""
    if config.session.backend == "none":
        return InMemoryLifecycleStore()
    base_dir = Path.cwd() if config_path is None else Path(config_path).parent
    path = Path(config.session.path or ".iris/session.db")
    if not path.is_absolute():
        path = base_dir / path
    return SQLiteStore(path.resolve())


__all__ = ["AgentRunner", "Clock"]
