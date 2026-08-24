"""Logical Agent run 的唯一 lifecycle owner。

``AgentRunner`` 是 Iris 对外唯一的 complete-run facade：它拥有 logical run 的创建、resume、
durable cancellation、settlement 观察、显式 recovery 与事件投递，``AgentRuntime`` 仅作为其
内部 engine 被驱动。所有 durable 事实都通过 ``LifecycleStore`` 的 aggregate command 提交，
进程内的 live resources（activation task、cancellation signal、deadline timer）不进入持久化。

Example:
    runner = AgentRunner.from_config_path("agent.yaml")
    result = await runner.start(AgentRunRequest(input="你好", session_id="default"))
    print(result.run.phase, result.assistant_message)
"""

# region imports
from __future__ import annotations

import asyncio
import inspect
import logging
import math
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

# endregion

logger = logging.getLogger(__name__)


def _supports_list_events_limit(store: LifecycleStore) -> bool:
    """判断 custom store 的 ``list_events`` 是否接受 keyword-only limit。"""
    try:
        parameters = inspect.signature(store.list_events).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "limit"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


class Clock(Protocol):
    """Runner 所需的 aware UTC wall-clock 边界。"""

    def now(self) -> datetime:
        """返回当前 aware 时间。"""


class _SystemClock:
    """默认 UTC wall clock。"""

    def now(self) -> datetime:
        return datetime.now(UTC)


class _MutableCancellationSignal(CancellationSignal):
    """一次 activation 共享的进程内 cooperative cancellation signal。

    同时承载显式 cancel 与 deadline 两个来源：``requested`` 对 runtime 表现一致，
    ``deadline_requested`` 单独保留原因，供 settlement 把 outcome 区分为
    ``cancelled`` 还是 ``deadline_exceeded``。
    """

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
    """当前进程中 activation 的 live resources；不进入持久化。

    只描述"此进程正在推进哪个 activation"，进程重启后一律靠 durable checkpoint 恢复，
    因此这里的字段都不需要序列化。

    Attributes:
        run_id (str): 所属 logical run id。
        activation_id (str): 该 activation 的 fence 标识，用于校验 store 侧 owner 未变。
        signal (_MutableCancellationSignal): 与 runtime 共享的协作式取消信号。
        task (asyncio.Task[RuntimeActivationResult] | None): 正在执行的 engine task。
        deadline_task (asyncio.Task[None] | None): absolute deadline 触发器。
        settled (asyncio.Event): activation 结算完成的进程内通知，供 cancel 观察者等待。
        events (list[RunEvent]): 本次 activation 累积的 durable events，settlement 后统一投递。
        steering (RuntimeSteeringPort | None): managed 组合层注入的安全边界 steering port。
        durable_event_callback (Callable[[RunEvent], None] | None): managed 组合层的同步 relay。
    """

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
    """创建 logical run 并把一次 engine activation 结算为 durable result。

    对外提供 complete-run 语义：``start()`` / ``resume()`` / ``recover()`` 的 coroutine 都在
    run 到达 waiting 或 terminal 后才返回。runner 是 durable ownership 的唯一持有者，engine
    只通过 ``StoreRuntimeCommitPort`` 提交事实；任何无法证明的 effect 都 fail closed 为
    ``outcome_unknown``。

    Attributes:
        runtime (AgentRuntime): 被驱动的内部 engine。
        store (LifecycleStore): 权威 durable store。
        observers (tuple[RunEventObserver, ...]): settlement 后 best-effort 的事件观察者。
        observer_event_timeout_s (float): 单个 observer event 的有限等待秒数。
        clock (Clock): aware UTC 时间源。
        interaction_service (HumanInteractionService): 无状态 HITL 领域服务。
        environment_fingerprint (str): 环境指纹，用于拒绝跨环境 resume/recover。

    Example:
        runner = AgentRunner.from_config_path("agent.yaml")
        result = await runner.start(AgentRunRequest(input="你好", session_id="default"))
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        store: LifecycleStore,
        observers: Sequence[RunEventObserver] = (),
        observer_event_timeout_s: float = 30.0,
        clock: Clock | None = None,
        interaction_service: HumanInteractionService | None = None,
    ) -> None:
        """绑定 engine、durable store 与观察者，装配唯一 lifecycle owner。

        ``observer_event_timeout_s`` 必须是有限正数；默认 30 秒。
        """
        if (
            isinstance(observer_event_timeout_s, bool)
            or not math.isfinite(observer_event_timeout_s)
            or observer_event_timeout_s <= 0
        ):
            raise ValueError("observer_event_timeout_s 必须是有限正数")
        self.runtime = runtime
        self.store = store
        self._store_list_events_supports_limit = _supports_list_events_limit(store)
        self.observers = tuple(observers)
        self.observer_event_timeout_s = observer_event_timeout_s
        self._observer_locks = tuple(asyncio.Lock() for _ in self.observers)
        self.clock = clock or _SystemClock()
        self.interaction_service = interaction_service or HumanInteractionService()
        # HITL 决定必须只落在 runner 的 aggregate transaction 里；带自有 store 的实现会
        # 造成第二条持久化路径，因此在装配期直接拒绝。
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
        observer_event_timeout_s: float = 30.0,
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
            observer_event_timeout_s=observer_event_timeout_s,
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
        observer_event_timeout_s: float = 30.0,
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
            observer_event_timeout_s=observer_event_timeout_s,
            clock=clock,
        )

    # endregion

    # ==========================================
    #              Run Lifecycle
    # ==========================================
    # region
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
        """创建 start activation，并注入可选的 process-local managed hooks。

        package-private，供 ``iris.harness`` 内部组合层（如 ``SessionManager``）使用。
        complete-run 语义与 ``start()`` 完全一致，只是额外接受三个进程内 hook。

        Args:
            request (AgentRunRequest): run 输入与 session 归属。
            options (AgentRunOptions | None): run 级限额与 runtime 选项。
            steering (RuntimeSteeringPort | None): activation-scoped 安全边界 steering port。
            durable_event_callback (Callable[[RunEvent], None] | None): durable event 同步 relay。
            activation_started (asyncio.Event | None): admission signal；仅在 create 已提交、
                events 已 relay 且 activation 已注册进 ``_active`` 后置位。

        Returns:
            RunResult: 到达 waiting 或 terminal 的 durable result。

        Raises:
            IrisRunConflictError: 当 create 返回的 initial checkpoint 与本地构造不一致时。
        """
        # --- 1. 构造 identity 与 initial checkpoint ---
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
        # --- 2. 提交 create 并处理立即 terminal ---
        # 立即 terminal（例如 create 时已超过 deadline）不进入 engine，也不产生 admission signal。
        events: list[RunEvent] = []
        self._record_events(events, created.events, durable_event_callback)
        if created.run.phase is RunPhase.TERMINAL:
            await self._deliver_events(events)
            return self._require_result(run_id)
        if created.checkpoint != checkpoint:
            raise IrisRunConflictError("create_run 返回了意外 initial checkpoint")

        # --- 3. 绑定 live resources 并推进 activation ---
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
        """创建 resume activation，并注入可选的 process-local managed hooks。

        package-private，语义与 ``resume()`` 一致，额外接受进程内组合 hook。

        Args:
            run_id (str): 目标 logical run id。
            interaction_id (str): 当前 pending 的 exact interaction id。
            response (HumanInteractionResponse): 人工决定。
            steering (RuntimeSteeringPort | None): activation-scoped 安全边界 steering port。
            durable_event_callback (Callable[[RunEvent], None] | None): durable event 同步 relay。
            activation_started (asyncio.Event | None): admission signal，语义见
                ``_start_managed``。

        Returns:
            RunResult: 到达 waiting 或 terminal 的 durable result。

        Raises:
            IrisRunNotFoundError: 当 run 或 interaction 不存在时。
            IrisRunConflictError: 当 interaction 归属、waiting identity 或 rebound cursor 不一致时。
            IrisRunStateError: 当 run phase 不允许 resume，或 commit 缺少必要事实时。
            IrisRunRecoveryError: 当 waiting run 缺少 checkpoint 或 checkpoint 校验失败时。
        """
        # --- 1. 校验 run/interaction identity ---
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

        # --- 2. 优先结算已到期的 waiting run ---
        # cancellation/deadline/interaction 过期都优先于人工响应，避免消费一个已作废的决定。
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

        # --- 3. 校验 checkpoint 并解决 interaction ---
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
        # --- 4. rebind 新 activation 并推进 ---
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
        # HITL 领域模型与 runtime 输入模型是两套边界类型，批准分支需要显式转换。
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
        """持久化首次 cancellation request，再 signal 当前进程 activation。

        ``cancellation_requested`` 只是 durable fact，不等于已取消：active run 需要等待协作式
        收口，waiting run 可在同一事务里直接结算为 terminal cancelled。重复请求是幂等的，
        只有首次会写入 durable 事实。

        Args:
            run_id (str): 目标 logical run id。
            reason (str | None): 取消原因，None 表示使用默认文案。

        Returns:
            RunSnapshot: 请求提交后的 run snapshot；terminal run 直接返回既有快照。

        Raises:
            IrisRunStateError: 当 run_id 或显式 reason 为空白时。
            IrisRunNotFoundError: 当 run 不存在时。
        """
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
        # 必须在 durable 请求落库之后才 signal，否则本地中断可能领先于持久化事实。
        # activation id 相等是 fence：跨进程或已换代的 activation 不受本进程 signal 影响。
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
        """请求取消并只观察 durable settlement；超时不写入新事实。

        同步且不协作的工具可能延迟 settlement，因此本方法不会提前返回 cancelled，只等待
        store 出现 terminal result。

        Args:
            run_id (str): 目标 logical run id。
            reason (str | None): 取消原因，None 表示使用默认文案。
            settlement_timeout (float | None): 观察超时秒数，必须大于 0；None 表示无限等待。

        Returns:
            RunResult: run 的 durable terminal result。

        Raises:
            IrisRunStateError: 当 settlement_timeout 非正数时。
            IrisRunNotFoundError: 当 run 不存在时。
            IrisRunObservationTimeoutError: 当超时仍未观察到 terminal result 时；该异常只表示
                观察失败，不改变任何 durable 事实。
        """
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
        """根据精确 activation fence 与 durable facts 显式恢复 run。

        只针对 active run 做 takeover：waiting run 应使用 ``resume()``，terminal run 退化为
        幂等读取。存在 unresolved claim 时绝不重放工具，而是把该 activation 结算为
        ``outcome_unknown``，因为已经发出的 effect 无法证明。

        Args:
            run_id (str): 目标 logical run id。
            expected_activation_id (str | None): active recovery 必须提供的 fence；用于确认
                要接管的正是调用方观察到的那一代 activation。

        Returns:
            RunResult: 到达 waiting 或 terminal 的 durable result。

        Raises:
            IrisRunNotFoundError: 当 run 不存在时。
            IrisRunConflictError: 当缺少 fence、fence 已变化或 rebound cursor 不一致时。
            IrisRunStateError: 当 waiting run 应走 resume，或本进程仍持有 live activation 时。
            IrisRunRecoveryError: 当 durable interaction/checkpoint 缺失或校验失败时。
        """
        # --- 1. 按 phase 分派 recovery 入口 ---
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
        # --- 2. 校验 active fence 与本地所有权 ---
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
        # --- 3. 由 durable facts 推导 recovery disposition ---
        # 任一 unresolved claim 都意味着 effect 结果不可证明，必须 fail closed，不看 checkpoint。
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
        # --- 4. 原子 abandon 旧 activation ---
        # 只有 RESUME 需要新一代 activation；FINALIZE 与 OUTCOME_UNKNOWN 都在同一事务里 terminal。
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

        # --- 5. 绑定新 activation 并继续推进 ---
        # recover 不是 managed 入口，因此不注入 steering / event callback / admission signal。
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
            # 只有 before_model/step 0 的输入尚未随 provider commit 进入 session history，
            # 需要从 durable request 重建；后续 checkpoint 再注入会造成重复输入。
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

    # endregion

    # ==========================================
    #         Recovery & Settlement Helpers
    # ==========================================
    # region
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
        # 已有 claim 说明工具 effect 可能正在发出，硬中断会让结果彻底不可证明；
        # 此时只留下 cooperative signal，让 runtime 自己走到可提交的边界。
        claimed = any(
            record.phase is ToolCallPhase.CLAIMED
            and record.claim_activation_id == active.activation_id
            for record in self.store.list_tool_calls(active.run_id)
        )
        task = active.task
        if claimed or task is None or task.done():
            return
        # deadline timer 可能来自其它线程的 loop，取消必须切回 task 所属 loop 执行。
        task.get_loop().call_soon_threadsafe(task.cancel)

    async def _observe_settlement(
        self,
        run_id: str,
        *,
        settlement_timeout: float | None,
    ) -> RunResult:
        """优先等待本地 settlement，并以纯读取覆盖跨进程 run。

        本进程持有 activation 时等待 ``settled`` 事件，否则退化为轮询 store，这样同一方法
        既能服务本地 run，也能观察其它进程正在推进的 run。"""
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
            # 即使有本地 settled 事件也保持有界等待，跨进程 settlement 只能靠重新读取发现。
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
        """在消费人工响应前验证 waiting checkpoint 的交叉事实。

        除了 recovery 同款的 run/session/usage/environment 校验，还要求 cursor 恰好停在
        该 interaction 对应的 tool call 上，并且 durable tool call 记录仍是 prepared。
        人工决定一旦被投影就会真实执行工具，因此必须先确认要执行的正是被批准的那一次调用。
        """
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
        """把对已 closed interaction 的重复提交解释为幂等读取。"""
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
        """把已到期的 waiting run 就地结算为 terminal，否则返回 None。

        waiting run 不占用 engine，只能在 resume/recover 等外部触点上判断到期。优先级为
        cancellation > deadline / interaction 过期；两者同时到期时取更早的时间点作为原因。
        """
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

    # endregion

    # ==========================================
    #               Durable Reads
    # ==========================================
    # region
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

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]:
        """读取 sequence 严格大于游标的 durable events。"""
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise IrisRunStateError("limit 必须是正整数", limit=limit)
        normalized = self._required_id(run_id)
        if limit is None:
            return self.store.list_events(normalized, after_sequence)
        if self._store_list_events_supports_limit:
            return self.store.list_events(normalized, after_sequence, limit=limit)
        return self.store.list_events(normalized, after_sequence)[:limit]

    # endregion

    # ==========================================
    #          Activation Settlement
    # ==========================================
    # region
    async def _run_activation(
        self,
        active: ActiveActivation,
        *,
        activation: RuntimeActivationInput,
        port: StoreRuntimeCommitPort,
    ) -> RunResult:
        """驱动一次 engine activation，并保证退出前形成 durable result。

        无论 engine 正常返回、被中断还是抛出未预期异常，都在此收敛为 terminal/waiting
        durable 事实；随后统一 revoke commit port、释放 live resources 并投递事件。

        Args:
            active (ActiveActivation): 本次 activation 的 live resources。
            activation (RuntimeActivationInput): 传给 engine 的 activation 输入。
            port (StoreRuntimeCommitPort): 该 activation 绑定的 commit port。

        Returns:
            RunResult: 到达 waiting 或 terminal 的 durable result。

        Raises:
            IrisRunConflictError: 当 durable 事实与 engine outcome 冲突时。
            IrisRunNotFoundError: 当 run 在 activation 期间消失时。
            IrisRunPersistenceError: 当 durable 写入失败时。
            IrisRunStateError: 当出现不可解释的 phase/outcome 组合，或缺少 durable result 时。
        """
        # --- 1. 启动 deadline timer 与 engine task ---
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
            # --- 2. 把每种退出路径映射为 durable outcome ---
            try:
                engine_result = await active.task
                self._settle_engine_result(active, engine_result, port)
            except asyncio.CancelledError:
                # 未经 signal 的取消来自外部调用方，不能被解释为 run 的 cancellation。
                if not active.signal.requested:
                    raise
                self._finish_cancelled_task(active, port)
            except (
                IrisRunConflictError,
                IrisRunNotFoundError,
                IrisRunPersistenceError,
                IrisRunStateError,
            ):
                # lifecycle 一致性错误说明 durable 事实已不可信，不再尝试写入 terminal。
                raise
            except Exception as exc:
                self._finish_unexpected(active, exc, port)
        # --- 3. 收口 live resources 并投递事件 ---
        finally:
            # 先 revoke 再释放资源，阻止迟到的 child 继续写入本 activation 的事实。
            port.revoke()
            try:
                await self._settle_live_resources(active)
            finally:
                # 只有仍属于自己的注册项才可摘除，避免误删已换代 activation。
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
        del port  # settlement 走 runner 自己的 finish_run，不复用已收口的 activation port。
        current = self.store.load_run(active.run_id)
        if current is None:
            raise IrisRunNotFoundError("run 在 cancellation 期间消失", run_id=active.run_id)
        # 工具 result 可能已在中断前正常提交并结算，此时不再覆盖既有 terminal。
        if current.phase is RunPhase.TERMINAL:
            return
        # 未提交的 claim 意味着 effect 不可证明，必须 fail closed 为 outcome unknown。
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
        """校验 engine outcome 与 durable 事实一致后结算 run。"""
        del port  # settlement 走 runner 自己的 finish_run，不复用已收口的 activation port。
        current = self.store.load_run(active.run_id)
        if current is None:
            raise IrisRunNotFoundError("run 在 activation 期间消失", run_id=active.run_id)
        # engine 的最终 cursor 必须已经落库，否则说明有 commit 丢失或被其它 activation 覆盖。
        checkpoint = self.store.load_checkpoint(active.run_id)
        if (
            checkpoint is None
            or RuntimeCursor.model_validate(checkpoint.engine_cursor) != result.cursor
        ):
            raise IrisRunConflictError("engine outcome cursor 与 durable checkpoint 不匹配")
        if current.phase is RunPhase.TERMINAL:
            return
        # 挂起由 commit port 在 suspend 事务中完成，runner 只做一致性确认。
        if result.outcome is RuntimeActivationOutcome.SUSPENDED:
            if current.phase is not RunPhase.WAITING:
                raise IrisRunStateError("suspended engine outcome 缺少 durable waiting state")
            return
        if current.phase is not RunPhase.ACTIVE:
            raise IrisRunStateError("non-suspended engine outcome 遇到非 active run")
        # engine 只感知统一的 cancellation signal，deadline 原因需要在此还原。
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
        """把 engine 的未预期异常结算为 failed 或 outcome unknown。"""
        del port  # settlement 走 runner 自己的 finish_run，不复用已收口的 activation port。
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
        """为配置了 absolute deadline 的 run 启动到期中断 timer。"""
        remaining = port.remaining_deadline_seconds()
        if remaining is None:
            return None

        async def request_at_deadline() -> None:
            await asyncio.sleep(remaining)
            self._interrupt_active(active, deadline=True)

        return asyncio.create_task(request_at_deadline())

    async def _settle_live_resources(self, active: ActiveActivation) -> None:
        """释放 activation 的进程内资源，确保没有悬挂 timer。"""
        if active.deadline_task is not None:
            active.deadline_task.cancel()
            with suppress(asyncio.CancelledError):
                await active.deadline_task
            active.deadline_task = None
        active.task = None

    async def _deliver_events(self, events: list[RunEvent]) -> None:
        """按 sequence 有序、去重地把 durable events 投递给全部 observer。

        Notes:
            投递是 best-effort：observer 抛出的异常只记录日志，不影响 durable 事实，也不
            中断后续 observer 与后续事件。
        """
        events_by_run: dict[str, dict[int, RunEvent]] = {}
        for event in events:
            events_by_run.setdefault(event.run_id, {}).setdefault(event.sequence, event)
        ordered = [
            event
            for run_events in events_by_run.values()
            for _, event in sorted(run_events.items())
        ]

        async def deliver_lane(
            observer: RunEventObserver,
            lock: asyncio.Lock,
        ) -> None:
            async with lock:
                for event in ordered:
                    try:
                        await asyncio.wait_for(
                            observer.on_event(event),
                            timeout=self.observer_event_timeout_s,
                        )
                    except TimeoutError:
                        logger.warning(
                            "observer event 超时",
                            extra={
                                "observer": type(observer).__qualname__,
                                "run_id": event.run_id,
                                "sequence": event.sequence,
                                "timeout_s": self.observer_event_timeout_s,
                            },
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning(
                            "run event observer 处理失败",
                            exc_info=True,
                            extra={
                                "observer": type(observer).__qualname__,
                                "run_id": event.run_id,
                                "sequence": event.sequence,
                            },
                        )

        await asyncio.gather(
            *(
                deliver_lane(observer, lock)
                for observer, lock in zip(self.observers, self._observer_locks, strict=True)
            )
        )

    # endregion

    # ==========================================
    #             Internal Helpers
    # ==========================================
    # region
    def _register(self, active: ActiveActivation, current_activation_id: str | None) -> None:
        """在 fence 校验通过后把 activation 登记为本进程 owner。"""
        if current_activation_id != active.activation_id:
            raise IrisRunConflictError("active map registration fence 不匹配")
        if active.run_id in self._active:
            raise IrisRunConflictError("run 已存在 process-local active activation")
        self._active[active.run_id] = active

    def _require_result(self, run_id: str) -> RunResult:
        """读取 settlement 后必然存在的 durable result。"""
        result = self.store.load_result(run_id)
        if result is None:
            raise IrisRunStateError("activation settlement 后缺少 durable result", run_id=run_id)
        return result

    def _now(self) -> datetime:
        """返回归一化到 UTC 的当前时间。"""
        now = self.clock.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise IrisRunStateError("runner clock 必须返回 aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _required_id(run_id: str) -> str:
        """规范化并校验非空 id。"""
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
        """去重收集 durable events，并同步隔离可选 callback。

        Notes:
            callback 异常只记录日志，不回滚已提交的 mutation，也不改变最终 ``RunResult``。
        """
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

    # endregion


def _build_lifecycle_store(
    config: AgentConfig,
    *,
    config_path: Path | None,
) -> LifecycleStore:
    """按 session 配置选择 harness-owned lifecycle store。

    Args:
        config (AgentConfig): 已校验的 agent 配置。
        config_path (Path | None): 配置文件路径；None 时以当前工作目录为相对路径基准。

    Returns:
        LifecycleStore: ``backend: none`` 返回内存 store，否则返回 SQLite store。
    """
    if config.session.backend == "none":
        return InMemoryLifecycleStore()
    base_dir = Path.cwd() if config_path is None else Path(config_path).parent
    path = Path(config.session.path or ".iris/session.db")
    if not path.is_absolute():
        path = base_dir / path
    return SQLiteStore(path.resolve())


__all__ = ["AgentRunner", "Clock"]
