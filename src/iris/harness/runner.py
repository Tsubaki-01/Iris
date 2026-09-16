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
import logging
import math
import uuid
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ..agents import AgentConfig, load_agent_config
from ..agents.config.subagent import load_subagent_catalog
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
from ..hitl.models import SubagentExpiryOwner
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
    RunControlSnapshot,
    RunErrorInfo,
    RunEvent,
    RunPhase,
    RunRecord,
    RunResult,
    RunSnapshot,
    RunStopReason,
    RunToolCallRecord,
    SessionSnapshot,
    ToolCallPhase,
    ToolErrorPolicy,
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
    RuntimeEventSink,
    RuntimeProvider,
    RuntimeSteeringPort,
    RuntimeStreamEvent,
)
from ..runtime._assembly import (
    RuntimeExecutionScope,
    SubagentAssembly,
    assemble_runtime,
    resolve_runtime_boundary,
)
from ..runtime.runtime import _project_tool_result_cursor, _tool_run_error
from ..store import InMemoryLifecycleStore, SQLiteStore
from ..tools import CancellationSignal, PermissionPolicy, ToolResult
from ..tools.subagent import ChildWaiting, SubagentExecutionOutcome, SubagentParentCall
from ._commit_port import StoreRuntimeCommitPort, _WaitingSubagentContinuationAdapter
from ._events import _RunEventCollector
from ._subagent import ChildProviderFactory, HarnessSubagentController
from .observer import RunEventObserver

if TYPE_CHECKING:
    from .streaming import LiveFact, LivePublisher

# endregion

logger = logging.getLogger(__name__)


class Clock(Protocol):
    """Runner 所需的 aware UTC wall-clock 边界。"""

    def now(self) -> datetime:
        """返回当前 aware 时间。"""


class _SystemClock:
    """默认 UTC wall clock。"""

    def now(self) -> datetime:
        return datetime.now(UTC)


class _RunnerPublisherRelay:
    """把 runtime sink 的 publish 调用交给 runner 的故障隔离边界。"""

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    def publish(self, fact: LiveFact) -> None:
        """委托 runner 发布一条 trusted fact。"""
        self._runner._publish_live_fact(fact)


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
        event_collector (_RunEventCollector): 与 commit port 共享的事件收集与同步 relay。
        steering (RuntimeSteeringPort | None): managed 组合层注入的安全边界 steering port。
    """

    run_id: str
    activation_id: str
    signal: _MutableCancellationSignal
    task: asyncio.Task[RuntimeActivationResult] | None = None
    deadline_task: asyncio.Task[None] | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    event_collector: _RunEventCollector = field(default_factory=_RunEventCollector)
    steering: RuntimeSteeringPort | None = None


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
        _live_publisher (LivePublisher | None): 可选的同进程 live fact publisher。
        clock (Clock): aware UTC 时间源。
        interaction_service (HumanInteractionService): 无状态 HITL 领域服务。

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
        live_publisher: LivePublisher | None = None,
    ) -> None:
        """绑定 engine、durable store 与观察者，装配唯一 lifecycle owner。

        ``observer_event_timeout_s`` 必须是有限正数；默认 30 秒。
        """
        if not 0 < observer_event_timeout_s < math.inf:
            raise ValueError("observer_event_timeout_s 必须是有限正数")
        self.runtime = runtime
        self.store = store
        self.observers = tuple(observers)
        self.observer_event_timeout_s = observer_event_timeout_s
        self._observer_locks = tuple(asyncio.Lock() for _ in self.observers)
        self.clock = clock or _SystemClock()
        self.interaction_service = interaction_service or HumanInteractionService()
        self._prepared = runtime.environment.mcp_manager is None
        self._closed = False
        self._live_publisher = live_publisher
        if live_publisher is None:
            self._stream_sink: RuntimeEventSink | None = None
        else:
            from .streaming import _RuntimeLiveSink

            self._stream_sink = _RuntimeLiveSink(_RunnerPublisherRelay(self))
        self._active: dict[str, ActiveActivation] = {}
        self._subagent_controller: HarnessSubagentController | None = None

    async def aprepare(self) -> None:
        """准备运行资源；失败后释放资源，必须新建 runner。"""
        if self._closed:
            raise IrisRunStateError("runner 已关闭")
        if self._prepared:
            return
        try:
            await self.runtime.environment.aprepare()
            self._prepared = True
        except BaseException:
            self._closed = True
            try:
                await self.runtime.environment.aclose()
            except Exception:
                logger.exception("MCP 准备失败后的资源关闭失败")
            raise

    async def aclose(self) -> None:
        """host 等原 start/resume/recover 完整结束后关闭自有环境资源。

        Raises:
            IrisRunStateError: 当前仍有 active activation，不能提前关闭资源。
        """
        if self._closed:
            return
        if self._active:
            raise IrisRunStateError("runner 仍有 active activation，不能关闭")
        self._closed = True
        await self.runtime.environment.aclose()

    @classmethod
    def from_config_path(
        cls,
        path: str | Path,
        *,
        provider: RuntimeProvider | None = None,
        permission_policy: PermissionPolicy | None = None,
        child_provider_factory: ChildProviderFactory | None = None,
        memory_service: MemoryService | None = None,
        store: LifecycleStore | None = None,
        observers: Sequence[RunEventObserver] = (),
        observer_event_timeout_s: float = 30.0,
        clock: Clock | None = None,
        api_key: str | None = None,
        live_publisher: LivePublisher | None = None,
    ) -> AgentRunner:
        """从 agent 配置路径装配 engine 与唯一 lifecycle store。"""
        config_path = Path(path)
        return cls.from_config(
            load_agent_config(config_path),
            config_path=config_path,
            provider=provider,
            permission_policy=permission_policy,
            child_provider_factory=child_provider_factory,
            memory_service=memory_service,
            store=store,
            observers=observers,
            observer_event_timeout_s=observer_event_timeout_s,
            clock=clock,
            api_key=api_key,
            live_publisher=live_publisher,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        config_path: Path | None = None,
        provider: RuntimeProvider | None = None,
        permission_policy: PermissionPolicy | None = None,
        child_provider_factory: ChildProviderFactory | None = None,
        memory_service: MemoryService | None = None,
        store: LifecycleStore | None = None,
        observers: Sequence[RunEventObserver] = (),
        observer_event_timeout_s: float = 30.0,
        clock: Clock | None = None,
        api_key: str | None = None,
        live_publisher: LivePublisher | None = None,
    ) -> AgentRunner:
        """从已校验配置装配 engine；durable ownership 只属于 harness。"""
        resolved_store = (
            store if store is not None else _build_lifecycle_store(config, config_path=config_path)
        )
        resolved_clock = clock if clock is not None else _SystemClock()
        boundary = resolve_runtime_boundary(
            config,
            config_path=config_path,
            permission_policy=permission_policy,
        )
        controller: HarnessSubagentController | None = None
        subagent: SubagentAssembly | None = None
        if config.tools.subagent is not None:
            base_dir = Path.cwd() if config_path is None else config_path.parent
            routes = load_subagent_catalog(base_dir / config.tools.subagent)
            controller = HarnessSubagentController(
                routes=routes,
                store=resolved_store,
                parent_boundary=boundary,
                child_provider_factory=child_provider_factory,
                clock=resolved_clock,
            )
            subagent = SubagentAssembly(routes, controller)
        runtime = assemble_runtime(
            config,
            config_path=config_path,
            provider=provider,
            memory_service=memory_service,
            api_key=api_key,
            execution_scope=RuntimeExecutionScope.ROOT,
            boundary=boundary,
            subagent=subagent,
        )
        runner = cls(
            runtime=runtime,
            store=resolved_store,
            observers=observers,
            observer_event_timeout_s=observer_event_timeout_s,
            clock=resolved_clock,
            live_publisher=live_publisher,
        )
        runner._subagent_controller = controller
        return runner

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
        await self.aprepare()
        command, cursor = self._build_start_facts(request, options=options)
        created = self.store.create_run(command)
        events = self._event_collector(durable_event_callback)
        events.record(created.events)
        if created.run.phase is RunPhase.TERMINAL:
            await self._deliver_events(events.events)
            return self._require_result(created.run.run_id)
        if created.checkpoint != command.initial_checkpoint:
            raise IrisRunConflictError("create_run 返回了意外 initial checkpoint")
        return await self._run_start_activation(
            created.run,
            activation_id=command.start_activation_id,
            cursor=cursor,
            events=events,
            steering=steering,
            activation_started=activation_started,
        )

    def _build_start_facts(
        self,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions | None = None,
    ) -> tuple[CreateRun, RuntimeCursor]:
        """为普通 start 与 child admission 构造同一组初始事实。"""
        resolved_options = options if options is not None else AgentRunOptions()
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
        )
        return CreateRun(
            request=resolved_request,
            options=resolved_options,
            agent_id=self.runtime.environment.agent_config.name,
            start_activation_id=activation_id,
            initial_checkpoint=checkpoint,
            now=self._now(),
        ), cursor

    async def _run_admitted_start(self, *, run_id: str, activation_id: str) -> RunResult:
        """驱动已由 parent admission 创建的 START，不再次 CreateRun。"""
        run = self.store.load_run(run_id)
        if run is None:
            raise IrisRunNotFoundError("admitted child run 不存在", run_id=run_id)
        events = self._event_collector()
        events.record(self.store.list_events(run_id))
        if run.phase is RunPhase.TERMINAL:
            await self._deliver_events(events.events)
            return self._require_result(run_id)
        checkpoint = self.store.load_checkpoint(run_id)
        if checkpoint is None:
            raise IrisRunRecoveryError("admitted child 缺少 checkpoint", run_id=run_id)
        cursor = RuntimeCursor.model_validate(checkpoint.engine_cursor)
        return await self._run_start_activation(
            run,
            activation_id=activation_id,
            cursor=cursor,
            events=events,
        )

    async def _run_start_activation(
        self,
        run: RunRecord,
        *,
        activation_id: str,
        cursor: RuntimeCursor,
        events: _RunEventCollector,
        steering: RuntimeSteeringPort | None = None,
        activation_started: asyncio.Event | None = None,
    ) -> RunResult:
        """共享 start live 资源注册；保留 managed admission signal 的原有顺序。"""
        active = ActiveActivation(
            run_id=run.run_id,
            activation_id=activation_id,
            signal=_MutableCancellationSignal(),
            event_collector=events,
            steering=steering,
        )
        port = StoreRuntimeCommitPort(
            workspace_root=self.runtime.environment.workspace_root,
            subagent_routes=self._subagent_controller.routes
            if self._subagent_controller is not None
            else None,
            store=self.store,
            run=run,
            activation_id=activation_id,
            cursor=cursor,
            clock=self._now,
            event_collector=events,
            interaction_service=self.interaction_service,
        )
        activation = RuntimeActivationInput(
            run_id=run.run_id,
            activation_id=activation_id,
            session_id=run.session_id,
            kind="start",
            run_input=run.request.input,
            initial_session_message_count=run.initial_session_message_count,
            cursor=cursor,
            options=run.options.runtime,
        )
        self._register(active, run.current_activation_id)
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
        needs_prepare = not self._prepared
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
        now = self._now()
        settled = await self._settle_waiting_if_due(
            run,
            interaction,
            now=now,
            steering=steering,
            durable_event_callback=durable_event_callback,
            activation_started=activation_started,
        )
        if settled is not None:
            return settled

        await self.aprepare()
        if needs_prepare:
            # 首次准备可能跨越 expiry 或其他 owner 的提交；重走原分派，只发生一次。
            return await self._resume_managed(
                run_id,
                interaction_id=interaction_id,
                response=response,
                steering=steering,
                durable_event_callback=durable_event_callback,
                activation_started=activation_started,
            )

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
            events = self._event_collector(durable_event_callback)
            events.record(resolved.events)
            if resolved.interaction is None:
                raise IrisRunStateError("resolve commit 缺少 interaction")
            interaction = resolved.interaction
        else:
            events = self._event_collector(durable_event_callback)
        # --- 4. rebind 新 activation 并推进 ---
        if interaction.request.subagent_origin is not None:
            return await self._resume_subagent_proxy(
                parent_run=run,
                proxy=interaction,
                parent_checkpoint=checkpoint,
                cursor=cursor,
                steering=steering,
                event_collector=events,
                activation_started=activation_started,
            )
        projection = self.interaction_service.project_response(interaction)
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
        events.record(begun.events)
        if begun.checkpoint is None:
            raise IrisRunStateError("begin activation 缺少 rebound checkpoint")
        if begun.checkpoint.engine_cursor != checkpoint.engine_cursor:
            raise IrisRunConflictError("rebound checkpoint cursor 与 waiting checkpoint 不匹配")
        # HITL 领域模型与 runtime 输入模型是两套边界类型，批准分支需要显式转换。
        runtime_projection = _runtime_interaction_projection(projection)
        active = ActiveActivation(
            run_id=run.run_id,
            activation_id=activation_id,
            signal=_MutableCancellationSignal(),
            event_collector=events,
            steering=steering,
        )
        port = StoreRuntimeCommitPort(
            workspace_root=self.runtime.environment.workspace_root,
            subagent_routes=self._subagent_controller.routes
            if self._subagent_controller is not None
            else None,
            store=self.store,
            run=begun.run,
            activation_id=activation_id,
            cursor=cursor,
            clock=self._now,
            event_collector=active.event_collector,
            interaction_service=self.interaction_service,
        )
        activation = RuntimeActivationInput(
            run_id=run.run_id,
            activation_id=activation_id,
            session_id=run.session_id,
            kind="resume",
            run_input=run.request.input,
            initial_session_message_count=run.initial_session_message_count,
            cursor=cursor,
            options=run.options.runtime,
            interaction_projection=runtime_projection,
        )
        self._register(active, begun.run.current_activation_id)
        if activation_started is not None:
            activation_started.set()
        return await self._run_activation(active, activation=activation, port=port)

    async def _resume_subagent_proxy(
        self,
        *,
        parent_run: RunRecord,
        proxy: HumanInteraction,
        parent_checkpoint: RunCheckpoint,
        cursor: RuntimeCursor,
        event_collector: _RunEventCollector,
        steering: RuntimeSteeringPort | None = None,
        activation_started: asyncio.Event | None = None,
    ) -> RunResult:
        """Response 已 durable 后推进 child，WAITING adapter 独占 parent rebind/finalize。"""
        controller = cast(HarnessSubagentController, self._subagent_controller)
        if activation_started is not None:
            activation_started.set()
        try:
            outcome = await controller.resume_proxy(parent_run=parent_run, proxy=proxy)
        except asyncio.CancelledError:
            current = cast(RunRecord, self.store.load_run(parent_run.run_id))
            if current.cancellation_requested_at is None:
                raise
            settled = await self._settle_waiting_if_due(
                current, proxy, now=self._now(), event_collector=event_collector
            )
            return cast(RunResult, settled)
        current = cast(RunRecord, self.store.load_run(parent_run.run_id))
        if current.phase is RunPhase.TERMINAL:
            await self._deliver_events(event_collector.events)
            return self._require_result(current.run_id)
        settled = await self._settle_waiting_if_due(
            current, proxy, now=self._now(), event_collector=event_collector
        )
        if settled is not None:
            return settled
        return await self._complete_subagent_proxy(
            parent_run=current,
            proxy=proxy,
            parent_checkpoint=parent_checkpoint,
            cursor=cursor,
            outcome=outcome,
            event_collector=event_collector,
            steering=steering,
        )

    async def _complete_subagent_proxy(
        self,
        *,
        parent_run: RunRecord,
        proxy: HumanInteraction,
        parent_checkpoint: RunCheckpoint,
        cursor: RuntimeCursor,
        outcome: SubagentExecutionOutcome,
        event_collector: _RunEventCollector,
        steering: RuntimeSteeringPort | None = None,
    ) -> RunResult:
        """正常回答与 child-owned 到期共用 WAITING rebind/finalize 后续编排。"""
        controller = cast(HarnessSubagentController, self._subagent_controller)
        adapter = _WaitingSubagentContinuationAdapter(
            store=self.store,
            interaction_service=self.interaction_service,
            clock=self._now,
            publish_live_fact=self._publish_live_fact,
            event_collector=event_collector,
            workspace_root=self.runtime.environment.workspace_root,
            routes=controller.routes,
        )
        call = SubagentParentCall(parent_run.run_id, proxy.tool_call_id)
        if isinstance(outcome, ChildWaiting):
            rebound = adapter.rebind(
                parent_run=parent_run, call=call, replaced_proxy=proxy, waiting=outcome
            )
            await self._deliver_events(event_collector.events)
            return cast(RunResult, rebound.result)
        result = self.runtime.environment.tool_bridge._normalize_subagent_result(
            cursor.tool_calls[cursor.next_tool_index],
            outcome,
            session_id=parent_run.session_id,
            run_id=parent_run.run_id,
            agent_id=parent_run.agent_id,
            workspace_root=self.runtime.environment.workspace_root,
            permission_mode=self.runtime.environment.agent_config.permissions.writes,
        )
        cursor_after = _project_tool_result_cursor(cursor, result, read_state=cursor.read_state)
        resumed = adapter.finalize(
            parent_run=parent_run,
            parent_checkpoint=parent_checkpoint,
            call=call,
            proxy=proxy,
            result=result,
            cursor_after=cursor_after,
        )
        if result.is_error and parent_run.options.runtime.tool_error_policy is ToolErrorPolicy.STOP:
            finished = self.store.finish_run(
                FinishRun(
                    run_id=parent_run.run_id,
                    expected_run_revision=resumed.run.revision,
                    activation_id=resumed.activation_id,
                    stop_reason=RunStopReason.FAILED,
                    assistant_message=cursor.assistant_message,
                    error=_tool_run_error(result),
                    now=self._now(),
                )
            )
            event_collector.record(finished.events)
            await self._deliver_events(event_collector.events)
            return self._require_result(parent_run.run_id)
        active = ActiveActivation(
            run_id=parent_run.run_id,
            activation_id=resumed.activation_id,
            signal=_MutableCancellationSignal(),
            event_collector=event_collector,
            steering=steering,
        )
        port = StoreRuntimeCommitPort(
            store=self.store,
            run=resumed.run,
            activation_id=resumed.activation_id,
            cursor=resumed.cursor,
            clock=self._now,
            event_collector=event_collector,
            interaction_service=self.interaction_service,
            workspace_root=self.runtime.environment.workspace_root,
            subagent_routes=controller.routes,
        )
        activation = RuntimeActivationInput(
            run_id=parent_run.run_id,
            activation_id=resumed.activation_id,
            session_id=parent_run.session_id,
            kind="resume",
            run_input=parent_run.request.input,
            initial_session_message_count=parent_run.initial_session_message_count,
            cursor=resumed.cursor,
            options=parent_run.options.runtime,
        )
        self._register(active, resumed.activation_id)
        return await self._run_activation(active, activation=activation, port=port)

    def request_cancel(
        self,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> RunSnapshot:
        """持久化首次 cancellation request，再 signal 当前进程 activation。

        ``cancellation_requested`` 只是 durable fact，不等于已取消。已有 claim 的 activation
        只接收 signal，由 executor 取消普通工具 body 并等待清理。普通 waiting run 可在同一
        事务里结算为 terminal cancelled；linked child 由 ``cancel()`` 继续结算。重复请求幂等，
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
                    settle_waiting=run.phase is RunPhase.WAITING
                    and not self._is_subagent_proxy(run),
                    now=self._now(),
                )
            )
            run = committed.run
            active = self._active.get(run.run_id)
            if active is not None:
                active.event_collector.record(committed.events)
            else:
                self._event_collector().record(committed.events)
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
        """请求取消、结算 linked child 并观察 durable settlement；观察超时不写 terminal。

        本方法等待 store 出现 terminal result，不保证原 ``start()`` / ``resume()`` task 已退出。
        慢 middleware、压住 ``CancelledError`` 的工具及 INLINE 阻塞仍可能延迟 settlement。
        需要等待 managed task 结束的 host 使用 ``SessionManager.close(cancel_run=True)``。

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
        budget = asyncio.timeout(settlement_timeout)
        try:
            async with budget:
                current = cast(RunRecord, self.store.load_run(normalized))
                await self._settle_linked_before_parent_stop(
                    parent_run=current, reason="parent cancelled"
                )
                current = cast(RunRecord, self.store.load_run(normalized))
                if current.phase is RunPhase.WAITING:
                    interaction = cast(
                        HumanInteraction,
                        self.store.load_interaction(current.pending_interaction_id or ""),
                    )
                    await self._settle_waiting_if_due(current, interaction, now=self._now())
                elif (
                    current.phase is RunPhase.ACTIVE
                    and normalized not in self._active
                    and self._subagent_controller is not None
                    and any(
                        self.store.load_subagent_link(normalized, tool.tool_call_id) is not None
                        for tool in self.store.list_tool_calls(normalized)
                        if tool.phase is ToolCallPhase.PREPARED
                    )
                ):
                    await self.recover(
                        normalized, expected_activation_id=current.current_activation_id
                    )
                return await self._observe_settlement(normalized, settlement_timeout=None)
        except TimeoutError as exc:
            if not budget.expired():
                raise
            raise IrisRunObservationTimeoutError(
                "等待 run cancellation settlement 超时", run_id=normalized
            ) from exc

    def _is_subagent_proxy(self, run: RunRecord) -> bool:
        """在 WAITING cancellation 操作边界识别需要 child-first 的 proxy。"""
        interaction = self.store.load_interaction(run.pending_interaction_id or "")
        return interaction is not None and interaction.request.subagent_origin is not None

    async def _settle_linked_before_parent_stop(
        self, *, parent_run: RunRecord, reason: str
    ) -> None:
        """停止 parent 前只结算当前 PREPARED 调用关联的 child。"""
        if self._subagent_controller is None:
            return
        for tool in self.store.list_tool_calls(parent_run.run_id):
            if tool.phase is ToolCallPhase.PREPARED:
                await self._subagent_controller.cancel_linked(
                    parent_run_id=parent_run.run_id,
                    parent_tool_call_id=tool.tool_call_id,
                    reason=reason,
                )

    async def recover(
        self,
        run_id: str,
        *,
        expected_activation_id: str | None = None,
    ) -> RunResult:
        """根据精确 activation fence 与 durable facts 显式恢复 run。

        Active run 需要精确 takeover fence；waiting 默认使用 ``resume()``，但已保存回答的
        Sub Agent proxy/outer gate 可直接恢复，已到期 waiting 则按 owner 结算。Terminal run
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
        needs_prepare = not self._prepared
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
            settled = await self._settle_waiting_if_due(run, interaction, now=self._now())
            if settled is not None:
                return settled
            if needs_prepare and self._prepared:
                return await self.recover(run_id, expected_activation_id=expected_activation_id)
            if (
                interaction.status is InteractionStatus.RESOLVED
                and interaction.request.tool_call.tool_name == "subagent"
            ):
                return await self._resume_managed(
                    run.run_id,
                    interaction_id=interaction.interaction_id,
                    response=cast(HumanInteractionResponse, interaction.response),
                )
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
            await self.aprepare()
            if needs_prepare:
                # 保留调用方 fence，重新读取 phase、checkpoint 与可能新出现的 claim。
                return await self.recover(run_id, expected_activation_id=expected_activation_id)
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
        recovered_events = self._event_collector()
        recovered_events.record(recovered.events)
        if recovered.run.phase is RunPhase.TERMINAL:
            await self._deliver_events(recovered_events.events)
            return self._require_result(run.run_id)
        if recovered.checkpoint is None or recovered_cursor is None or new_activation_id is None:
            raise IrisRunRecoveryError("recover commit 缺少 rebound activation facts")
        if recovered.checkpoint.engine_cursor != recovered_cursor.model_dump(mode="json"):
            raise IrisRunConflictError("recover rebound checkpoint cursor 已变化")

        # --- 5. 绑定新 activation 并继续推进 ---
        # recover 不是 managed 入口，因此不注入 steering / event callback / admission signal。
        active = ActiveActivation(
            run_id=run.run_id,
            activation_id=new_activation_id,
            signal=_MutableCancellationSignal(),
            event_collector=recovered_events,
        )
        port = StoreRuntimeCommitPort(
            workspace_root=self.runtime.environment.workspace_root,
            subagent_routes=self._subagent_controller.routes
            if self._subagent_controller is not None
            else None,
            store=self.store,
            run=recovered.run,
            activation_id=new_activation_id,
            cursor=recovered_cursor,
            clock=self._now,
            event_collector=active.event_collector,
            interaction_service=self.interaction_service,
        )
        activation = RuntimeActivationInput(
            run_id=run.run_id,
            activation_id=new_activation_id,
            session_id=run.session_id,
            kind="recover",
            interaction_projection=self._stored_interaction_projection(
                recovered.run, recovered_cursor
            ),
            run_input=run.request.input,
            initial_session_message_count=run.initial_session_message_count,
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
    def _stored_interaction_projection(
        self,
        run: RunRecord,
        cursor: RuntimeCursor,
    ) -> RuntimeApprovedToolCall | ToolResult | None:
        """恢复当前工具已有的 HITL 回答；linked subagent 由 controller 继续。"""
        if cursor.position != "tool_batch":
            return None
        call = cursor.tool_calls[cursor.next_tool_index]
        if (
            call.name == "subagent"
            and self.store.load_subagent_link(run.run_id, call.id) is not None
        ):
            return None
        record = self.store.load_tool_call(run.run_id, call.id)
        if record is None or record.interaction_id is None:
            return None
        interaction = self.store.load_interaction(record.interaction_id)
        if (
            interaction is not None
            and interaction.status in {InteractionStatus.RESOLVED, InteractionStatus.CLOSED}
            and interaction.response is not None
        ):
            return _runtime_interaction_projection(
                self.interaction_service.project_response(interaction)
            )
        return None

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

        除了 recovery 同款的 run/session/usage 校验，还要求 cursor 恰好停在
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
        current_record = self.store.load_tool_call(run.run_id, subject.tool_call_id)
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

    async def _settle_waiting_if_due(
        self,
        run: RunRecord,
        interaction: HumanInteraction,
        *,
        now: datetime,
        steering: RuntimeSteeringPort | None = None,
        durable_event_callback: Callable[[RunEvent], None] | None = None,
        activation_started: asyncio.Event | None = None,
        event_collector: _RunEventCollector | None = None,
    ) -> RunResult | None:
        """把已到期的 waiting run 就地结算为 terminal，否则返回 None。

        waiting run 不占用 engine，只能在 resume/recover 等外部触点上判断到期。优先级为
        cancellation > deadline / interaction 过期；两者同时到期时取更早的时间点作为原因。
        """
        origin = interaction.request.subagent_origin
        deadline = run.options.limits.deadline_at
        child_owned = origin is not None and origin.expiry_owner in {
            SubagentExpiryOwner.CHILD_INTERACTION_EXPIRY,
            SubagentExpiryOwner.CHILD_EFFECTIVE_DEADLINE,
            SubagentExpiryOwner.OUTER_TOOL_TIMEOUT,
        }
        if (
            child_owned
            and interaction.status is InteractionStatus.PENDING
            and interaction.expires_at is not None
            and now >= interaction.expires_at
            and run.cancellation_requested_at is None
            and (deadline is None or now < deadline)
        ):
            needs_prepare = not self._prepared
            await self.aprepare()
            if needs_prepare:
                current = cast(RunRecord, self.store.load_run(run.run_id))
                if current.phase is RunPhase.TERMINAL:
                    return self._require_result(run.run_id)
                if (
                    current.phase is not RunPhase.WAITING
                    or current.pending_interaction_id != interaction.interaction_id
                ):
                    raise IrisRunConflictError("MCP 准备期间 waiting identity 已变化")
                current_interaction = cast(
                    HumanInteraction, self.store.load_interaction(interaction.interaction_id)
                )
                return await self._settle_waiting_if_due(
                    current,
                    current_interaction,
                    now=self._now(),
                    steering=steering,
                    durable_event_callback=durable_event_callback,
                    activation_started=activation_started,
                    event_collector=event_collector,
                )
            checkpoint = self.store.load_checkpoint(run.run_id)
            if checkpoint is None:
                raise IrisRunRecoveryError("waiting run 缺少 durable checkpoint", run_id=run.run_id)
            cursor = self._validate_resume_checkpoint(run, interaction, checkpoint)
            if activation_started is not None:
                activation_started.set()
            controller = cast(HarnessSubagentController, self._subagent_controller)
            outcome = await controller.expire_proxy(parent_run=run, proxy=interaction)
            current = cast(RunRecord, self.store.load_run(run.run_id))
            if current.cancellation_requested_at is not None or (
                deadline is not None and self._now() >= deadline
            ):
                return await self._settle_waiting_if_due(current, interaction, now=self._now())
            return await self._complete_subagent_proxy(
                parent_run=current,
                proxy=interaction,
                parent_checkpoint=checkpoint,
                cursor=cursor,
                outcome=outcome,
                event_collector=event_collector or self._event_collector(durable_event_callback),
                steering=steering,
            )
        stop_reason: RunStopReason | None = None
        close_reason: str | None = None
        if run.cancellation_requested_at is not None:
            stop_reason = RunStopReason.CANCELLED
            close_reason = "cancelled"
        else:
            expiry = (
                interaction.expires_at
                if interaction.status is InteractionStatus.PENDING and not child_owned
                else None
            )
            if deadline is not None and expiry is not None and now >= deadline and now >= expiry:
                stop_reason = (
                    RunStopReason.DEADLINE_EXCEEDED
                    if deadline <= expiry
                    else RunStopReason.INTERACTION_EXPIRED
                )
                close_reason = stop_reason.value
            elif deadline is not None and now >= deadline:
                stop_reason = RunStopReason.DEADLINE_EXCEEDED
                close_reason = "deadline_exceeded"
            elif expiry is not None and now >= expiry:
                stop_reason = RunStopReason.INTERACTION_EXPIRED
                close_reason = "interaction_expired"
        if stop_reason is None:
            return None
        event_cursor = run.last_event_sequence
        if activation_started is not None:
            activation_started.set()
        await self._settle_linked_before_parent_stop(parent_run=run, reason=stop_reason.value)
        run = cast(RunRecord, self.store.load_run(run.run_id))
        if run.phase is RunPhase.TERMINAL:
            if event_collector is not None:
                await self._deliver_events(event_collector.events)
            return self._require_result(run.run_id)
        if run.cancellation_requested_at is not None:
            stop_reason = RunStopReason.CANCELLED
            close_reason = "cancelled"
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
        events = event_collector or self._event_collector(durable_event_callback)
        events.record(self.store.list_events(run.run_id, event_cursor))
        await self._deliver_events(events.events)
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

    def get_run_control(self, run_id: str) -> RunControlSnapshot:
        """读取 run 的窄控制快照，不加载完整请求、选项和模型输出。"""
        control = self.store.load_run_control(self._required_id(run_id))
        if control is None:
            raise IrisRunNotFoundError("run 不存在", run_id=run_id)
        return control

    def get_session(self, session_id: str) -> SessionSnapshot:
        """读取 exact runner store 中的 session durable snapshot。"""
        normalized = session_id.strip()
        if not normalized:
            raise IrisRunStateError("session_id 不能为空")
        return self.store.load_session(normalized)

    def get_result(self, run_id: str) -> RunResult | None:
        """读取 waiting/terminal durable result；active run 返回 ``None``。"""
        normalized = self._required_id(run_id)
        if self.store.load_run(normalized) is None:
            raise IrisRunNotFoundError("run 不存在", run_id=run_id)
        return self.store.load_result(normalized)

    def list_tool_calls(self, run_id: str) -> list[RunToolCallRecord]:
        """读取一个已存在 logical run 的全部 durable tool calls。"""
        run = self.get_run(run_id)
        return self.store.list_tool_calls(run.run_id)

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]:
        """读取 sequence 严格大于游标的 durable events。"""
        return self.store.list_events(
            self._required_id(run_id),
            after_sequence,
            limit=limit,
        )

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
                    stream_sink=self._stream_sink,
                )
            )
            # --- 2. 把每种退出路径映射为 durable outcome ---
            try:
                engine_result = await active.task
                if engine_result.outcome is not RuntimeActivationOutcome.SUSPENDED:
                    await self._settle_linked_before_parent_stop(
                        parent_run=port.run, reason="parent execution finished"
                    )
                self._settle_engine_result(active, engine_result, port)
            except asyncio.CancelledError:
                # 未经 signal 的取消来自外部调用方，不能被解释为 run 的 cancellation。
                if not active.signal.requested:
                    raise
                await self._settle_linked_before_parent_stop(
                    parent_run=port.run, reason="parent interrupted"
                )
                self._finish_cancelled_task(active, port)
            except (
                IrisRunConflictError,
                IrisRunNotFoundError,
                IrisRunPersistenceError,
                IrisRunRecoveryError,
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
            await self._deliver_events(active.event_collector.events)
        return self._require_result(active.run_id)

    def _finish_cancelled_task(
        self,
        active: ActiveActivation,
        port: StoreRuntimeCommitPort,
    ) -> None:
        """把被中断的 async operation 映射为可证明的 durable outcome。"""
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
        active.event_collector.record(committed.events)

    def _settle_engine_result(
        self,
        active: ActiveActivation,
        result: RuntimeActivationResult,
        port: StoreRuntimeCommitPort,
    ) -> None:
        """校验 engine outcome 与 durable 事实一致后结算 run。"""
        current = self.store.load_run(active.run_id)
        if current is None:
            raise IrisRunNotFoundError("run 在 activation 期间消失", run_id=active.run_id)
        # engine 的最终 cursor 必须已经落库，否则说明有 commit 丢失或被其它 activation 覆盖。
        checkpoint = self.store.load_checkpoint(active.run_id)
        if port.cursor != result.cursor or checkpoint != port.checkpoint:
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
        # 失败先于 timer 获得调度时，仍以同一 Clock 的 absolute deadline 确认到期原因。
        outcome = result.outcome
        if outcome is RuntimeActivationOutcome.FAILED:
            deadline_at = current.options.limits.deadline_at
            if deadline_at is not None and self._now() >= deadline_at:
                active.signal.request_deadline()
        if active.signal.deadline_requested and outcome in {
            RuntimeActivationOutcome.CANCELLED,
            RuntimeActivationOutcome.FAILED,
        }:
            self._finish_cancelled_task(active, port)
            return
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
        active.event_collector.record(committed.events)

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
        active.event_collector.record(committed.events)

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

    def _event_collector(
        self,
        durable_event_callback: Callable[[RunEvent], None] | None = None,
    ) -> _RunEventCollector:
        """构造一次事件收集 owner，绑定 managed callback 与 live publisher。"""
        return _RunEventCollector(self._compose_durable_event_callback(durable_event_callback))

    def _publish_live_fact(self, fact: LiveFact) -> None:
        """Best-effort 发布 runner fact，不影响 runtime 或 durable settlement。"""
        publisher = self._live_publisher
        if publisher is None:
            return
        if isinstance(fact, RunEvent):
            fact_kind = fact.kind.value
            run_id = fact.run_id
            session_id = fact.session_id
            activation_id = fact.activation_id
        elif isinstance(fact, RuntimeStreamEvent):
            fact_kind = fact.kind
            run_id = fact.run_id
            session_id = fact.session_id
            activation_id = fact.activation_id
        else:
            fact_kind = f"submission.{fact.event.state}"
            run_id = fact.event.run_id
            session_id = fact.session_id
            activation_id = None
        try:
            publisher.publish(fact)
        except Exception:
            logger.warning(
                "live publisher 处理 runner fact 失败",
                extra={
                    "publisher": type(publisher).__qualname__,
                    "fact_kind": fact_kind,
                    "run_id": run_id,
                    "session_id": session_id,
                    "activation_id": activation_id,
                },
                exc_info=True,
            )

    def _compose_durable_event_callback(
        self,
        callback: Callable[[RunEvent], None] | None,
    ) -> Callable[[RunEvent], None] | None:
        """组合既有 callback 与 publisher，供 commit port relay 新事件。"""
        if callback is None and self._live_publisher is None:
            return None

        def relay(event: RunEvent) -> None:
            try:
                if callback is not None:
                    callback(event)
            finally:
                self._publish_live_fact(event)

        return relay

    # endregion


def _runtime_interaction_projection(
    projection: ApprovedToolCall | ToolResult,
) -> RuntimeApprovedToolCall | ToolResult:
    """将 HITL 批准投影到 runtime 当前契约，工具结果直接复用。"""
    if isinstance(projection, ApprovedToolCall):
        return RuntimeApprovedToolCall.model_construct(
            interaction_id=projection.interaction_id,
            tool_call_id=projection.tool_call_id,
            tool_name=projection.tool_name,
            fingerprint=projection.fingerprint,
        )
    return projection


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
