"""Sub Agent 的 shared-store child 编排与最小工具结果投影。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..agents import AgentConfig, load_agent_config
from ..exceptions import (
    IrisConfigError,
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from ..hitl.models import (
    HumanInteraction,
    HumanInteractionResponse,
    InteractionStatus,
    SubagentExpiryOwner,
    SubagentProxyOrigin,
)
from ..lifecycle.models import (
    AgentRunOptions,
    AgentRunRequest,
    RunPhase,
    RunRecord,
    RunResult,
    RunStopReason,
)
from ..lifecycle.store import AdmitChildRun, LifecycleStore
from ..message import TextBlock
from ..runtime._assembly import (
    RuntimeAssemblyBoundary,
    RuntimeExecutionScope,
    assemble_runtime,
    resolve_runtime_boundary,
)
from ..runtime.environment import RuntimeProvider
from ..tools import ToolErrorInfo, ToolResult
from ..tools.subagent import (
    ChildWaiting,
    SubagentExecutionOutcome,
    SubagentInvocation,
    SubagentRoute,
    SubagentRouteTable,
)

if TYPE_CHECKING:
    from .runner import AgentRunner, Clock


class ChildProviderFactory(Protocol):
    """按选中 child 的普通配置构造独立 provider。"""

    def __call__(self, config: AgentConfig, *, config_path: Path) -> RuntimeProvider:
        """接收已加载的 child 配置，构造独立 provider。

        Args:
            config: 被选择的普通 child AgentConfig。
            config_path: Child YAML 的声明路径，供 provider 工厂解析相关资源。

        Returns:
            RuntimeProvider: 本次 child 执行使用的 provider。
        """
        ...


class HarnessSubagentController:
    """用普通 child Runner 驱动唯一 durable child，不拥有第二套 lifecycle。"""

    def __init__(
        self,
        *,
        routes: SubagentRouteTable,
        store: LifecycleStore,
        parent_boundary: RuntimeAssemblyBoundary,
        child_provider_factory: ChildProviderFactory | None,
        clock: Clock,
    ) -> None:
        self.routes = routes
        self.store = store
        self.parent_boundary = parent_boundary
        self.child_provider_factory = child_provider_factory
        self.clock = clock
        self._live_children: dict[str, tuple[AgentRunner, asyncio.Task[RunResult]]] = {}

    async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
        """Fresh admission 或按 exact link 继续，返回 child WAITING/TERMINAL。"""
        parent_call = invocation.parent_call
        parent = self._load_run(parent_call.parent_run_id)
        link = self.store.load_subagent_link(
            parent_call.parent_run_id, parent_call.parent_tool_call_id
        )
        route = invocation.call.route
        if link is not None:
            return await self._continue_linked(
                route, parent, link.child_run_id, parent_call.parent_tool_call_id
            )
        try:
            runner = self._assemble_child(route)
        except IrisConfigError as exc:
            code = (
                "SUBAGENT_WORKSPACE_DISJOINT"
                if "parent_workspace" in exc.context
                else "SUBAGENT_CONFIG_ERROR"
            )
            return _error_result(route.selector, code, "Selected child configuration is invalid")
        request = AgentRunRequest.model_construct(
            input=invocation.call.prompt,
            session_id=f"session_{uuid.uuid4().hex}",
            run_id=f"run_{uuid.uuid4().hex}",
            metadata={},
        )
        child_create, _ = runner._build_start_facts(request, options=AgentRunOptions())
        tool = self.store.load_tool_call(parent.run_id, parent_call.parent_tool_call_id)
        if tool is None:
            raise IrisRunNotFoundError(
                "parent tool call 不存在", tool_call_id=parent_call.parent_tool_call_id
            )
        link = self.store.admit_child_run(
            AdmitChildRun(
                parent_run_id=parent.run_id,
                expected_parent_run_revision=parent.revision,
                parent_activation_id=cast(str, parent.current_activation_id),
                parent_tool_call_id=tool.tool_call_id,
                expected_parent_tool_version=tool.version,
                child_create=child_create,
            )
        )
        if link.child_run_id != child_create.request.run_id:
            return await self._continue_linked(
                route, parent, link.child_run_id, parent_call.parent_tool_call_id
            )
        return await self._run_child(
            route,
            parent,
            self._load_run(link.child_run_id),
            parent_call.parent_tool_call_id,
            runner,
            lambda: runner._run_admitted_start(
                run_id=link.child_run_id, activation_id=child_create.start_activation_id
            ),
        )

    def _assemble_child(self, route: SubagentRoute) -> AgentRunner:
        """只加载 selected ordinary config，直接消费 CHILD boundary 与独立 provider。"""
        from .runner import AgentRunner

        config = load_agent_config(route.config_path)
        boundary = resolve_runtime_boundary(
            config, config_path=route.config_path, parent_boundary=self.parent_boundary
        )
        provider = (
            None
            if self.child_provider_factory is None
            else self.child_provider_factory(
                config,
                config_path=route.config_path,
            )
        )
        runtime = assemble_runtime(
            config,
            config_path=route.config_path,
            provider=provider,
            memory_service=None,
            api_key=None,
            execution_scope=RuntimeExecutionScope.CHILD,
            boundary=boundary,
        )
        return AgentRunner(runtime=runtime, store=self.store, clock=self.clock)

    async def resume_proxy(
        self,
        *,
        parent_run: RunRecord,
        proxy: HumanInteraction,
    ) -> SubagentExecutionOutcome:
        """从已保存 response 和 durable selector 继续原 child，不重新 admission。"""
        origin = cast(SubagentProxyOrigin, proxy.request.subagent_origin)
        link = self.store.load_subagent_link(parent_run.run_id, proxy.tool_call_id)
        if link is None or link.child_run_id != origin.child_run_id:
            raise IrisRunConflictError("proxy 与 exact child link 不匹配")
        route = self.routes.routes.get(origin.agent_selector)
        if route is None:
            raise IrisRunRecoveryError(
                "proxy selector 不在当前 catalog", selector=origin.agent_selector
            )
        child = self._load_run(link.child_run_id)
        if child.phase is RunPhase.WAITING:
            if child.pending_interaction_id == origin.child_interaction_id:
                runner = self._assemble_child(route)
                return await self._run_child(
                    route,
                    parent_run,
                    child,
                    proxy.tool_call_id,
                    runner,
                    lambda: runner.resume(
                        child.run_id,
                        interaction_id=origin.child_interaction_id,
                        response=cast(HumanInteractionResponse, proxy.response),
                    ),
                )
            previous = self.store.load_interaction(origin.child_interaction_id)
            if (
                previous is None
                or previous.status is not InteractionStatus.CLOSED
                or previous.response != proxy.response
            ):
                raise IrisRunConflictError("child 当前 interaction 无法由 proxy response 解释")
        return await self._continue_linked(route, parent_run, child.run_id, proxy.tool_call_id)

    async def _continue_linked(
        self, route: SubagentRoute, parent: RunRecord, child_run_id: str, parent_tool_call_id: str
    ) -> SubagentExecutionOutcome:
        """ACTIVE 走普通 recover；WAITING 读取或按到期 owner 结算，TERMINAL 只读。"""
        child = self._load_run(child_run_id)
        if child.phase is RunPhase.ACTIVE:
            runner = self._assemble_child(route)
            return await self._run_child(
                route,
                parent,
                child,
                parent_tool_call_id,
                runner,
                lambda: runner.recover(
                    child_run_id, expected_activation_id=child.current_activation_id
                ),
            )
        else:
            result = self.store.load_result(child_run_id)
            if result is None:
                raise IrisRunStateError("linked child 缺少 durable result", run_id=child_run_id)
        projected = self._project_outcome(route.selector, parent, child, result)
        if (
            isinstance(projected, ChildWaiting)
            and projected.proxy_expires_at is not None
            and self.clock.now() >= projected.proxy_expires_at
        ):
            return await self._expire_child(
                parent_run_id=parent.run_id,
                parent_tool_call_id=parent_tool_call_id,
                child_run_id=child.run_id,
                selector=route.selector,
                owner=projected.expiry_owner,
            )
        return projected

    async def _run_child(
        self,
        route: SubagentRoute,
        parent: RunRecord,
        child: RunRecord,
        parent_tool_call_id: str,
        runner: AgentRunner,
        operation: Callable[[], Coroutine[Any, Any, RunResult]],
    ) -> SubagentExecutionOutcome:
        """本次调用持有 child task；absolute timer 与取消都先完成 child settlement。"""
        deadline = _outer_deadline(parent, child)
        # WAITING continuation 没有 parent activation timer，仍需在 child await 期间响应 deadline。
        parent_deadline = parent.options.limits.deadline_at
        if parent.phase is RunPhase.WAITING and parent_deadline is not None:
            deadline = parent_deadline if deadline is None else min(deadline, parent_deadline)
        remaining = None if deadline is None else (deadline - self.clock.now()).total_seconds()
        if remaining is not None and remaining <= 0:
            await self.cancel_linked(
                parent_run_id=parent.run_id,
                parent_tool_call_id=parent_tool_call_id,
                reason="subagent timeout",
            )
            return _error_result(
                route.selector, "SUBAGENT_TIMEOUT", "Child agent timed out", child.run_id
            )
        task = asyncio.create_task(operation())
        self._live_children[child.run_id] = (runner, task)
        try:
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except TimeoutError:
                await self.cancel_linked(
                    parent_run_id=parent.run_id,
                    parent_tool_call_id=parent_tool_call_id,
                    reason="subagent timeout",
                )
                return _error_result(
                    route.selector, "SUBAGENT_TIMEOUT", "Child agent timed out", child.run_id
                )
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(
                    self.cancel_linked(
                        parent_run_id=parent.run_id,
                        parent_tool_call_id=parent_tool_call_id,
                        reason="parent execution interrupted",
                    )
                )
                # 重复取消不能释放仍在 settlement 中的原 child ownership。
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
                raise
            return self._project_outcome(
                route.selector, parent, self._load_run(child.run_id), result
            )
        finally:
            self._live_children.pop(child.run_id, None)

    async def cancel_linked(
        self, *, parent_run_id: str, parent_tool_call_id: str, reason: str
    ) -> None:
        """Exact child 先收到普通取消；本地等待原 task，非本地 ACTIVE 使用原 fence 恢复。"""
        link = self.store.load_subagent_link(parent_run_id, parent_tool_call_id)
        if link is None:
            return
        child = self._load_run(link.child_run_id)
        live = self._live_children.get(child.run_id)
        if live is not None:
            runner, task = live
            runner.request_cancel(child.run_id, reason=reason)
            await asyncio.shield(task)
        elif child.phase is not RunPhase.TERMINAL:
            tool = self.store.load_tool_call(parent_run_id, parent_tool_call_id)
            route = self.routes.routes[
                cast(str, tool.arguments.get("agent") or self.routes.default)
            ]
            runner = self._assemble_child(route)
            cancelled = runner.request_cancel(child.run_id, reason=reason)
            if cancelled.phase is RunPhase.ACTIVE:
                await runner.recover(
                    child.run_id, expected_activation_id=cancelled.current_activation_id
                )

    async def expire_proxy(self, *, parent_run: RunRecord, proxy: HumanInteraction) -> ToolResult:
        """消费 durable child-owned expiry；不伪造回答或改写 canonical owner。"""
        origin = cast(SubagentProxyOrigin, proxy.request.subagent_origin)
        return await self._expire_child(
            parent_run_id=parent_run.run_id,
            parent_tool_call_id=proxy.tool_call_id,
            child_run_id=origin.child_run_id,
            selector=origin.agent_selector,
            owner=origin.expiry_owner,
        )

    async def _expire_child(
        self,
        *,
        parent_run_id: str,
        parent_tool_call_id: str,
        child_run_id: str,
        selector: str,
        owner: SubagentExpiryOwner | None,
    ) -> ToolResult:
        """ACTIVE recovery 与 durable proxy 共用 child 到期执行。"""
        if owner in {
            SubagentExpiryOwner.OUTER_TOOL_TIMEOUT,
            SubagentExpiryOwner.PARENT_RUN_DEADLINE,
            SubagentExpiryOwner.PARENT_INTERACTION_TIMEOUT,
        }:
            await self.cancel_linked(
                parent_run_id=parent_run_id,
                parent_tool_call_id=parent_tool_call_id,
                reason="subagent timeout",
            )
        else:
            child = self._load_run(child_run_id)
            if child.phase is not RunPhase.TERMINAL:
                runner = self._assemble_child(self.routes.routes[selector])
                await runner.recover(
                    child.run_id, expected_activation_id=child.current_activation_id
                )
        return _error_result(selector, "SUBAGENT_TIMEOUT", "Child agent timed out", child_run_id)

    def _load_run(self, run_id: str) -> RunRecord:
        """读取必须存在的 durable run；缺失不创建替代。"""
        run = self.store.load_run(run_id)
        if run is None:
            raise IrisRunNotFoundError("Sub Agent run 不存在", run_id=run_id)
        return run

    def _project_outcome(
        self,
        selector: str,
        parent: RunRecord,
        child: RunRecord,
        result: RunResult,
    ) -> SubagentExecutionOutcome:
        """仅投影等待所需事实或最终 assistant 文本，usage 留在 child。"""
        if child.phase is RunPhase.WAITING:
            interaction = result.pending_interaction
            interaction_timeout = parent.options.limits.interaction_timeout_seconds
            candidates = [
                (parent.options.limits.deadline_at, SubagentExpiryOwner.PARENT_RUN_DEADLINE),
                (
                    None
                    if interaction_timeout is None
                    else self.clock.now() + timedelta(seconds=interaction_timeout),
                    SubagentExpiryOwner.PARENT_INTERACTION_TIMEOUT,
                ),
                (interaction.expires_at, SubagentExpiryOwner.CHILD_INTERACTION_EXPIRY),
                (child.options.limits.deadline_at, SubagentExpiryOwner.CHILD_EFFECTIVE_DEADLINE),
                (
                    _outer_deadline(parent, child),
                    SubagentExpiryOwner.OUTER_TOOL_TIMEOUT,
                ),
            ]
            present: list[tuple[datetime, SubagentExpiryOwner]] = [
                (time, owner) for time, owner in candidates if time is not None
            ]
            expires_at, owner = min(present, key=lambda item: item[0]) if present else (None, None)
            return ChildWaiting(child.run_id, interaction, expires_at, owner)
        if result.run.stop_reason is RunStopReason.COMPLETED:
            if result.assistant_message is not None:
                return ToolResult(
                    tool_use_id="",
                    tool_name="subagent",
                    content=[TextBlock(text=result.assistant_message.text)],
                    metadata={"agent_selector": selector, "child_run_id": child.run_id},
                )
            return _error_result(
                selector,
                "SUBAGENT_FAILED",
                "Child agent completed without an assistant message",
                child.run_id,
            )
        code, message = {
            RunStopReason.FAILED: ("SUBAGENT_FAILED", "Child agent failed"),
            RunStopReason.BUDGET_EXHAUSTED: (
                "SUBAGENT_BUDGET_EXHAUSTED",
                "Child agent exhausted its model-step budget",
            ),
            RunStopReason.DEADLINE_EXCEEDED: ("SUBAGENT_TIMEOUT", "Child agent timed out"),
            RunStopReason.INTERACTION_EXPIRED: ("SUBAGENT_TIMEOUT", "Child agent timed out"),
            RunStopReason.CANCELLED: ("SUBAGENT_CANCELLED", "Child agent was cancelled"),
            RunStopReason.OUTCOME_UNKNOWN: (
                "SUBAGENT_OUTCOME_UNKNOWN",
                "Child agent outcome is unknown",
            ),
        }[cast(RunStopReason, result.run.stop_reason)]
        return _error_result(selector, code, message, child.run_id)


def _error_result(
    selector: str, code: str, message: str, child_run_id: str | None = None
) -> ToolResult:
    """构造最小、不可自动重试的模型可见 child 失败。"""
    metadata = {"agent_selector": selector}
    if child_run_id is not None:
        metadata["child_run_id"] = child_run_id
    return ToolResult(
        tool_use_id="",
        tool_name="subagent",
        is_error=True,
        error=ToolErrorInfo(code=code, message=message, retryable=False),
        metadata=metadata,
    )


def _outer_deadline(parent: RunRecord, child: RunRecord) -> datetime | None:
    """Outer budget 只以 durable child admission 时间为 anchor。"""
    timeout = parent.options.runtime.tool_timeout_seconds
    return None if timeout is None else child.created_at + timedelta(seconds=timeout)
