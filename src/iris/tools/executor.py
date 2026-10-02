"""工具执行入口。

控制工具逻辑的核心调用执行器模块，负责分发单次和并发批次执行，
捕获异常将其归一化为标准化返回结构，以及拦截并交由策略验证权限。

Example:
    executor = ToolExecutor(registry)
    results = await executor.execute_many(blocks, context)
"""

# region imports
from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from ..command.models import CommandStatus, CommandStopSlot
from ..command.service import CommandBinding
from ..exceptions import (
    IrisCancellationRequestedError,
    IrisToolExecutionError,
    IrisToolNotFoundError,
    IrisToolValidationError,
)
from ..hitl.models import (
    HumanInteractionRequest,
    PermissionPrompt,
    QuestionPrompt,
    ToolCallSnapshot,
    make_call_fingerprint,
)
from ..hooks.dispatcher import HookDispatcher
from ..hooks.models import ToolAfterEvent, ToolBeforeEvent
from ..message import ToolUseBlock
from ._execution_control import ToolExecutionControlSlot
from ._hooks import consume_tool_hook_control, drain_hook_commands
from ._io import run_tool_io
from ._middleware_chain import CONTROL_ERRORS, drain_operation, run_middleware_chain
from ._read_state import ReadFileState
from .artifacts import artifact_store_for, truncate_tool_result
from .base import (
    BaseTool,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
    ToolTimeoutOwner,
)
from .circuit import CircuitBreaker
from .middleware import ToolCall, ToolMiddleware, ToolNext
from .permissions import (
    DefaultPermissionPolicy,
    PermissionDecision,
    PermissionEffect,
    PermissionPolicy,
)
from .registry import ToolRegistry
from .subagent import (
    ChildWaiting,
    SubagentCallInput,
    SubagentExecutionOutcome,
    SubagentParentCall,
    SubagentTool,
)


class ToolEffectGuard(Protocol):
    """工具进入 middleware/body/artifact 生命周期前的 required guard。"""

    def before_effect(self, prepared: PreparedToolCall) -> None:
        """仅在 durable effect claim 成功后返回。"""


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """无副作用预检后的一条工具调用计划。"""

    tool_use: ToolUseBlock
    tool: BaseTool | None = None
    validated_input: BaseModel | dict[str, Any] | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    permission: PermissionDecision | None = None
    human_request: HumanInteractionRequest | None = None
    preflight_result: ToolResult | None = None
    timeout_owner: ToolTimeoutOwner = ToolTimeoutOwner.RUNTIME


@dataclass(frozen=True, slots=True)
class ToolBatchPlan:
    """保持原始顺序的一组工具预检结果。"""

    calls: tuple[PreparedToolCall, ...] = ()

    @property
    def first_human_gate(self) -> PreparedToolCall | None:
        return next((call for call in self.calls if call.human_request is not None), None)


@dataclass(slots=True)
class _ToolExecutionFacts:
    """包装器不能改写的真实 body 事实与最后有效下游结果。"""

    body_result: ToolResult | None = None
    last_result: ToolResult | None = None
    revealed_tools: tuple[str, ...] | None = None


# endregion


class ToolExecutor:
    """执行 `ToolUseBlock` 的工具执行器。

    充当模型产生的工具调用原语向实际实现的桥梁，具备参数校验、
    上下文透传、结果加工等防腐层功能。自动将所有不稳定的底层异常进行标准化拦截。

    Attributes:
        registry (ToolRegistry): 工具注册表，用以按名拾取工具。
        permission_policy (PermissionPolicy): 用于执行前权限风控检测卡控。

    Example:
        executor = ToolExecutor(registry)
        res = await executor.execute_one(tool_use_block, ctx)
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_policy: PermissionPolicy | None = None,
        middleware: Sequence[ToolMiddleware] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        hook_dispatcher: HookDispatcher | None = None,
        command_binding: CommandBinding | None = None,
    ) -> None:
        """初始化执行器。

        配置注册表、策略及长输出阈值等核心工作设置。

        Args:
            registry (ToolRegistry): 配置好可用方法的当前状态总库。
            permission_policy (PermissionPolicy | None): 安全及交互式授权拦截规则处理器。
            middleware (Sequence[ToolMiddleware] | None): 工具调用生命周期钩子。
            circuit_breaker (CircuitBreaker | None): 连续失败熔断器。
            hook_dispatcher (HookDispatcher | None): 本 Agent 共享的工具 Hook 派发器。
            command_binding (CommandBinding | None): Hook 停止事实收口使用的命令依赖。
        """
        self.registry = registry
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.middleware = list(middleware or [])
        self.circuit_breaker = circuit_breaker
        self.hook_dispatcher = hook_dispatcher
        self.command_binding = command_binding

    # endregion

    # ==========================================
    #               Execute Methods
    # ==========================================
    # region
    async def execute_one(
        self,
        tool_use: ToolUseBlock,
        context: ToolExecutionContext,
        *,
        approved_tool_call_id: str | None = None,
    ) -> ToolResult:
        """执行单个工具调用并将错误归一化为 `ToolResult`。

        提供对调用全周期（取用、拦截、加工、执行与存盘）的集成支撑，
        将一切错误包装在不打断模型运行的专用响应结构中返回。

        Args:
            tool_use (ToolUseBlock): 携带名字、唯一标记与输入参数的模型指令结构。
            context (ToolExecutionContext): 生命周期环境，提供跨工具所需的文件及会话根环境。

        Returns:
            ToolResult: 带有确切结果与成功与否标识体，支持大体积落盘记录。
        """
        prepared = self._prepare_call(tool_use, context)
        return await self._execute_current(
            prepared,
            context,
            approved_tool_call_id=approved_tool_call_id,
        )

    async def execute_many(
        self,
        tool_uses: Sequence[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """执行多个工具调用，只读且并发安全的连续批次并发执行。

        识别指令清单，自动收集处于可以只读的工具做并行调度从而节约周期；
        遇到写状态改变等风险工具立刻退化为阻塞调用以保证顺序。

        Args:
            tool_uses (Sequence[ToolUseBlock]): 从模型抽取的当次循环执行清单集。
            context (ToolExecutionContext): 供当次工具使用的透传依赖。

        Returns:
            list[ToolResult]: 所有结果合包，结果序列一一对应清单集原有次序。
        """
        results: list[ToolResult] = []
        batch: list[PreparedToolCall] = []
        for tool_use in tool_uses:
            prepared = self._prepare_call(tool_use, context)
            if self._is_read_only_concurrency_safe(prepared):
                batch.append(prepared)
                continue
            if batch:
                results.extend(await self._execute_read_batch(batch, context))
                batch = []
            call_context = _copy_context_for_parallel_call(context)
            results.append(await self._execute_current(prepared, call_context))
            if context.read_state is None:
                context.read_state = call_context.read_state
        if batch:
            results.extend(await self._execute_read_batch(batch, context))
        return results

    def prepare_many(
        self,
        tool_uses: Sequence[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> ToolBatchPlan:
        """在不触发执行生命周期的情况下预检一批工具调用。"""
        calls = [self._prepare_call(tool_use, context) for tool_use in tool_uses]
        return ToolBatchPlan(calls=tuple(calls))

    async def execute_prepared(
        self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
        *,
        approved_tool_call_id: str | None = None,
        effect_guard: ToolEffectGuard | None = None,
    ) -> ToolResult:
        """刷新当前权限后执行一条已完成输入校验的调用。"""
        current = self._refresh_permission(prepared, context)
        return await self._execute_current(
            current,
            context,
            approved_tool_call_id=approved_tool_call_id,
            effect_guard=effect_guard,
        )

    def prepare_subagent_continuation(
        self, tool_use: ToolUseBlock, context: ToolExecutionContext
    ) -> PreparedToolCall:
        """恢复已关联 child 或 outer response 的 raw 输入，不重复 outer permission。"""
        return self._prepare_input(tool_use)

    async def execute_subagent_prepared(
        self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
        *,
        parent_call: SubagentParentCall,
        approved_tool_call_id: str | None = None,
        linked_continuation: bool = False,
    ) -> SubagentExecutionOutcome:
        """专用委派入口；不进入 middleware、breaker 或 parent effect claim。"""
        current = prepared if linked_continuation else self._refresh_permission(prepared, context)
        permission_error = self._permission_error(
            current,
            approved_tool_call_id=approved_tool_call_id,
        )
        if permission_error is not None:
            return permission_error
        tool = cast(SubagentTool, current.tool)
        params = cast(SubagentCallInput, current.validated_input)
        context.call_id = current.tool_use.id
        context.tool_name = current.tool_use.name
        if context.cancellation is not None:
            context.cancellation.raise_if_requested()
        outcome = await tool.execute_subagent(params, context, parent_call=parent_call)
        if isinstance(outcome, ChildWaiting):
            return outcome
        # Controller 的 lifecycle/recovery 错误不属于模型可见工具失败。
        return await self._finalize_result(
            tool_use=current.tool_use, tool=tool, result=outcome, context=context
        )

    async def _finalize_result(
        self,
        *,
        tool_use: ToolUseBlock,
        tool: BaseTool,
        result: ToolResult,
        context: ToolExecutionContext,
        revealed_tools: tuple[str, ...] | None = None,
    ) -> ToolResult:
        """普通工具与 child 共用最终归一化及 artifact 失败投影。"""
        try:
            return await run_tool_io(
                lambda: self._normalize_result_identity_and_artifact(
                    tool_use=tool_use,
                    tool=tool,
                    result=result,
                    context=context,
                    revealed_tools=revealed_tools,
                )
            )
        except IrisToolExecutionError as exc:
            code, message = _tool_error_code_and_message(exc.message, allow_structured=True)
            return self._error_result(
                tool_use, code, message, max_chars=tool.definition.max_result_chars
            )

    async def _execute_current(
        self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
        *,
        approved_tool_call_id: str | None = None,
        effect_guard: ToolEffectGuard | None = None,
    ) -> ToolResult:
        """执行当前阶段已完成 lookup、校验与鉴权的调用。"""
        permission_error = self._permission_error(
            prepared,
            approved_tool_call_id=approved_tool_call_id,
        )
        if permission_error is not None:
            return permission_error
        return await self._execute_authorized(
            prepared,
            context,
            effect_guard=effect_guard,
        )

    # endregion

    # ==========================================
    #               Helper Methods
    # ==========================================
    # region
    def _prepare_call(
        self,
        tool_use: ToolUseBlock,
        context: ToolExecutionContext,
    ) -> PreparedToolCall:
        """查找、校验并鉴权一条调用，不触发执行生命周期。"""
        return self._refresh_permission(self._prepare_input(tool_use), context)

    def _prepare_input(self, tool_use: ToolUseBlock) -> PreparedToolCall:
        """共享 lookup/raw validation，不拥有权限裁决。"""
        tool: BaseTool | None = None
        try:
            tool = self.registry.get(tool_use.name)
            validated_input = tool.validate_input(tool_use.input)
            arguments = (
                validated_input.model_dump()
                if isinstance(validated_input, BaseModel)
                else dict(validated_input)
            )
            return PreparedToolCall(
                tool_use=tool_use,
                tool=tool,
                validated_input=validated_input,
                arguments=arguments,
                timeout_owner=tool.timeout_owner,
            )
        except IrisToolNotFoundError:
            return PreparedToolCall(
                tool_use=tool_use,
                preflight_result=self._error_result(
                    tool_use,
                    "NOT_FOUND",
                    f"工具不存在: {tool_use.name}",
                ),
            )
        except (IrisToolValidationError, ValidationError) as exc:
            return PreparedToolCall(
                tool_use=tool_use,
                tool=tool,
                preflight_result=self._error_result(
                    tool_use,
                    "VALIDATION_ERROR",
                    str(exc),
                    max_chars=tool.definition.max_result_chars if tool else None,
                ),
            )
        except Exception as exc:
            return PreparedToolCall(
                tool_use=tool_use,
                tool=tool,
                preflight_result=self._error_result(
                    tool_use,
                    "PERMISSION_ERROR",
                    str(exc),
                    max_chars=tool.definition.max_result_chars if tool else None,
                ),
            )

    def _refresh_permission(
        self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> PreparedToolCall:
        """只刷新已校验调用的权限与 HITL 投影。"""
        tool = prepared.tool
        validated_input = prepared.validated_input
        if tool is None or validated_input is None:
            return prepared
        try:
            decision = self.permission_policy.check(tool, prepared.arguments, context)
            preflight_result = self._permission_preflight_result(
                prepared.tool_use,
                tool,
                decision,
            )
            human_request = (
                None
                if preflight_result is not None
                else _human_interaction_request(
                    prepared.tool_use,
                    tool,
                    validated_input,
                    prepared.arguments,
                    decision,
                    context,
                )
            )
            return replace(
                prepared,
                permission=decision,
                human_request=human_request,
                preflight_result=preflight_result,
            )
        except Exception as exc:
            return replace(
                prepared,
                permission=None,
                human_request=None,
                preflight_result=self._error_result(
                    prepared.tool_use,
                    "PERMISSION_ERROR",
                    str(exc),
                    max_chars=tool.definition.max_result_chars,
                ),
            )

    def _permission_preflight_result(
        self,
        tool_use: ToolUseBlock,
        tool: BaseTool,
        decision: PermissionDecision,
    ) -> ToolResult | None:
        """把当前 permission decision 投影为预检错误。"""
        if decision.effect is PermissionEffect.DENY:
            return self._error_result(
                tool_use,
                "PERMISSION_ERROR",
                decision.reason,
                max_chars=tool.definition.max_result_chars,
                details={
                    "require_confirmation": False,
                    "effect": decision.effect.value,
                    **decision.metadata,
                },
            )
        if tool.definition.group == "human" and decision.effect is PermissionEffect.REQUIRE_HUMAN:
            return self._error_result(
                tool_use,
                "PERMISSION_ERROR",
                "human interaction tool 不能同时要求额外人工授权",
                max_chars=tool.definition.max_result_chars,
                details={
                    "require_confirmation": True,
                    "effect": decision.effect.value,
                    **decision.metadata,
                },
            )
        return None

    def _permission_error(
        self,
        prepared: PreparedToolCall,
        *,
        approved_tool_call_id: str | None,
    ) -> ToolResult | None:
        """按安全优先级将当前鉴权结果映射为错误。"""
        if prepared.preflight_result is not None:
            return prepared.preflight_result
        decision = prepared.permission
        max_chars = prepared.tool.definition.max_result_chars if prepared.tool else None
        if decision is not None and decision.effect is PermissionEffect.DENY:
            return self._error_result(
                prepared.tool_use,
                "PERMISSION_ERROR",
                decision.reason,
                max_chars=max_chars,
                details={
                    "require_confirmation": False,
                    "effect": decision.effect.value,
                    **decision.metadata,
                },
            )
        if prepared.human_request is not None and isinstance(
            prepared.human_request.prompt,
            QuestionPrompt,
        ):
            return self._error_result(
                prepared.tool_use,
                "HITL_REQUIRED",
                "human interaction 必须由 runtime 处理",
                max_chars=max_chars,
            )
        if (
            decision is not None
            and decision.effect is PermissionEffect.REQUIRE_HUMAN
            and approved_tool_call_id != prepared.tool_use.id
        ):
            return self._error_result(
                prepared.tool_use,
                "PERMISSION_ERROR",
                decision.reason,
                max_chars=max_chars,
                details={
                    "require_confirmation": True,
                    "effect": decision.effect.value,
                    **decision.metadata,
                },
            )
        return None

    async def _execute_authorized(
        self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
        *,
        effect_guard: ToolEffectGuard | None = None,
    ) -> ToolResult:
        """执行已通过当前鉴权的调用及其生命周期。"""
        from .discovery import ToolSearchTool

        tool_use = prepared.tool_use
        tool = cast(BaseTool, prepared.tool)
        validated_input = cast(BaseModel | dict[str, Any], prepared.validated_input)
        context.call_id = tool_use.id
        context.tool_name = tool_use.name
        if self.circuit_breaker is not None:
            try:
                self.circuit_breaker.before_call(tool.name)
            except IrisToolExecutionError as exc:
                return self._error_result(
                    tool_use,
                    str(exc.context.get("code", "CIRCUIT_OPEN")),
                    exc.message,
                    details=exc.context,
                    max_chars=tool.definition.max_result_chars,
                )
        if context.cancellation is not None:
            context.cancellation.raise_if_requested()
        if effect_guard is not None:
            effect_guard.before_effect(prepared)
        if context.cancellation is not None:
            context.cancellation.raise_if_requested()
        facts = _ToolExecutionFacts()
        control = ToolExecutionControlSlot()
        hook_feedback: tuple[str, ...] = ()

        def remember_result(result: ToolResult) -> None:
            """只有真实 body 已知，包装器的下游结果才可用于延迟控制提交。"""
            if facts.body_result is not None:
                facts.last_result = result

        async def leaf() -> ToolResult:
            """保留 body 原异常给包装器，同时独立记录真实成功或失败。"""
            try:
                result = await self._run_tool_body(tool, validated_input, context)
            except CONTROL_ERRORS as error:
                control.record(error)
                raise
            except Exception as error:
                facts.body_result = self._execution_error_result(tool_use, tool, error)
                facts.last_result = facts.body_result
                self._record_breaker_result(tool.name, facts.body_result)
                raise
            facts.body_result = result
            facts.last_result = result
            if isinstance(tool, ToolSearchTool) and not result.is_error:
                facts.revealed_tools = tuple(result.metadata["context_revealed_tools"])
            self._record_breaker_result(tool.name, result)
            return result

        async def middleware_operation() -> ToolResult:
            """空链直接执行 body，否则组合一次调用的包装层。"""
            if not self.middleware:
                return await leaf()
            call = ToolCall(
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                arguments=deepcopy(prepared.arguments),
                agent_id=context.agent_id,
                session_id=context.session_id,
                run_id=context.metadata.get("run_id"),
                activation_id=context.metadata.get("activation_id"),
                workspace_root=context.workspace_root,
            )
            return await run_middleware_chain(
                self.middleware,
                call,
                leaf,
                control=control,
                remember_result=remember_result,
                middleware_error=lambda error: self._error_result(
                    tool_use, "MIDDLEWARE_ERROR", str(error)
                ),
            )

        async def operation() -> ToolResult:
            """同一调用 owner 中依次执行 before、包装链与有资格的 after。"""
            nonlocal hook_feedback
            dispatcher = self.hook_dispatcher
            event_fields = {
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "run_id": context.metadata.get("run_id"),
                "activation_id": context.metadata.get("activation_id"),
                "workspace": str(context.workspace_root),
                "call_id": tool_use.id,
                "tool_name": tool_use.name,
                "arguments": prepared.arguments,
            }
            if dispatcher is not None:
                before = await dispatcher.dispatch(
                    ToolBeforeEvent(**event_fields), cancellation=context.cancellation
                )
                if before.control is not None:
                    await consume_tool_hook_control(
                        before.control, context, self.command_binding, body_known=False
                    )
                if before.rejection is not None:
                    return self._error_result(
                        tool_use, before.rejection.code, before.rejection.reason
                    )
            try:
                result = await middleware_operation()
            except CONTROL_ERRORS:
                raise
            except Exception as error:
                result = self._execution_error_result(tool_use, tool, error)
            # 该字段只属于本轮真实 Hook；缓存或包装器返回值不能制造反馈。
            result = result.model_copy(update={"hook_feedback": ()})
            remember_result(result)
            slot = context.command_stop_slot
            if (
                dispatcher is not None
                and facts.body_result is not None
                and control.control is None
                and slot.cleanup_error is None
                and slot.status
                not in {CommandStatus.CANCELLED, CommandStatus.ENVIRONMENT_INTERRUPTED}
                and not (context.cancellation is not None and context.cancellation.requested)
                and (slot.receipt is None or slot.status is CommandStatus.TIMED_OUT)
            ):
                # 普通 body timeout 可触发 after，但其命令必须先排空。
                if slot.receipt is not None:
                    await drain_hook_commands(context, self.command_binding)
                snapshot = result.model_copy(
                    update={"tool_use_id": tool_use.id, "tool_name": tool_use.name}
                )
                after = await dispatcher.dispatch(
                    ToolAfterEvent(
                        **event_fields,
                        result=snapshot,
                        body_status="error" if facts.body_result.is_error else "success",
                    ),
                    cancellation=context.cancellation,
                )
                hook_feedback = after.feedback
                result = result.model_copy(update={"hook_feedback": hook_feedback})
                remember_result(result)
                if after.control is not None:
                    await consume_tool_hook_control(
                        after.control, context, self.command_binding, body_known=True
                    )
            return result

        try:
            result = await self._run_call_operation(operation, context, control)
        except CONTROL_ERRORS as error:
            control.record(error)
            if facts.last_result is None or context.execution_control is None:
                raise control.control.error from None
            result = facts.last_result
        except Exception as error:
            result = self._execution_error_result(tool_use, tool, error)

        if control.control is not None:
            if facts.last_result is None or context.execution_control is None:
                raise control.control.error
            context.execution_control.record(control.control.error)
            result = facts.last_result

        # 所有 effect 后的结果在同一出口保存；落盘失败只返回错误，不重复尝试写入。
        result = await self._finalize_result(
            tool_use=tool_use,
            tool=tool,
            result=result.model_copy(update={"hook_feedback": hook_feedback}),
            context=context,
            revealed_tools=facts.revealed_tools,
        )
        return result

    def _execution_error_result(
        self, tool_use: ToolUseBlock, tool: BaseTool, error: Exception
    ) -> ToolResult:
        """只在执行 owner 将未恢复的普通异常投影为工具错误。"""
        if isinstance(error, (IrisToolValidationError, ValidationError)):
            return self._error_result(tool_use, "VALIDATION_ERROR", str(error))
        if isinstance(error, IrisToolExecutionError):
            code, message = _tool_error_code_and_message(
                error.message,
                allow_structured=(
                    tool.definition.group == "file" or error.message.startswith("ARTIFACT_ERROR:")
                ),
            )
            return self._error_result(tool_use, code, message)
        return self._error_result(tool_use, "EXECUTION_ERROR", str(error))

    async def _run_call_operation(
        self,
        operation: ToolNext,
        context: ToolExecutionContext,
        control: ToolExecutionControlSlot,
    ) -> ToolResult:
        """监控整个包装生命周期；claim 后仅 signal 改变也能中断后置等待。"""
        task = asyncio.create_task(operation())
        cancellation = context.cancellation
        cancellation_sent = False
        try:
            while not task.done():
                if cancellation is not None and cancellation.requested:
                    control.record(IrisCancellationRequestedError("工具调用响应 activation 取消"))
                    if not cancellation_sent:
                        task.cancel()
                        cancellation_sent = True
                await asyncio.wait(
                    {task},
                    timeout=0.01 if cancellation is not None and not cancellation_sent else None,
                )
        except asyncio.CancelledError as error:
            control.record(error)
            if not task.done() and not cancellation_sent:
                task.cancel()
            await drain_operation(task)
        if cancellation is not None and cancellation.requested:
            control.record(IrisCancellationRequestedError("工具调用响应 activation 取消"))
        return task.result()

    async def _run_tool_body(
        self,
        tool: BaseTool,
        validated_input: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """body 启动前检查最新信号；整条调用的取消及排空由外层唯一拥有。"""
        if context.cancellation is not None:
            context.cancellation.raise_if_requested()
        return await tool.arun(validated_input, context)

    def _normalize_result_identity_and_artifact(
        self,
        *,
        tool_use: ToolUseBlock,
        tool: BaseTool,
        result: ToolResult,
        context: ToolExecutionContext,
        revealed_tools: tuple[str, ...] | None,
    ) -> ToolResult:
        """绑定当前 identity，并对最终交付正文执行 ordinary artifact 归一化。"""
        metadata = {
            key: value
            for key, value in result.metadata.items()
            if key not in {"context_retention", "context_tool_name", "context_revealed_tools"}
        }
        if isinstance(metadata.get("extra"), dict):
            metadata["extra"] = {
                key: value
                for key, value in metadata["extra"].items()
                if key != "context_revealed_tools"
            }
        if revealed_tools is not None and not result.is_error:
            metadata["context_revealed_tools"] = list(revealed_tools)
        # 框架事实最后加入，to_block_metadata 归一化时覆盖用户 extra 中的同名字段。
        metadata["context_retention"] = tool.definition.context_retention
        metadata["context_tool_name"] = tool.definition.name
        normalized = result.model_copy(
            update={
                "tool_use_id": tool_use.id,
                "tool_name": tool_use.name,
                "metadata": metadata,
            }
        )
        return artifact_store_for(
            context,
            preview_chars=tool.definition.preview_chars,
            preview_mode=tool.definition.preview_mode,
        ).persist_if_large(
            normalized,
            max_chars=tool.definition.max_result_chars,
        )

    def _error_result(
        self,
        tool_use: ToolUseBlock,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
        max_chars: int | None = None,
    ) -> ToolResult:
        """构造错误工具结果。

        作为包装错误栈、异常文本到统一响应对象出口的方法，保证模型侧认知规范。

        Args:
            tool_use (ToolUseBlock): 引发异常的原请求信息。
            code (str): 大写带下划线的标准类型标记。
            message (str): 描述异常情形的详细反馈体。
            details (dict[str, object] | None): 附加可能存在的部分详细风控上下文。
            max_chars: 预检短路的正文预算；省略时由 effect 后的最终处理保存完整错误。

        Returns:
            ToolResult: is_error 生效情况下的专供结构体。
        """
        result = ToolResult(
            tool_use_id=tool_use.id,
            tool_name=tool_use.name,
            is_error=True,
            error=ToolErrorInfo(code=code, message=message, details=details or {}),
        )
        if max_chars is not None and len(result.model_content) > max_chars:
            return truncate_tool_result(
                result,
                max_chars=max_chars,
                preview_chars=max_chars,
                suffix="\n[错误说明已截断]",
            )
        return result

    async def _execute_read_batch(
        self,
        prepared_calls: list[PreparedToolCall],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """并发执行连续只读批次。

        将一串经过判断安全的调用批量压入异步调度池内争取快速返回。

        Args:
            prepared_calls (list[PreparedToolCall]): 需要并发执行的已准备调用。
            context (ToolExecutionContext): 生命周期环境。

        Returns:
            list[ToolResult]: 生成的已完成数据流集。
        """
        if context.read_state is None and self._has_file_tool(prepared_calls):
            context.read_state = ReadFileState()
        tasks = (
            self._execute_current(prepared, _copy_context_for_parallel_call(context))
            for prepared in prepared_calls
        )
        return list(await asyncio.gather(*tasks))

    def _has_file_tool(self, prepared_calls: list[PreparedToolCall]) -> bool:
        """判断批次中是否包含内置文件工具。"""
        return any(
            prepared.tool is not None and prepared.tool.definition.group == "file"
            for prepared in prepared_calls
        )

    def _is_read_only_concurrency_safe(self, prepared: PreparedToolCall) -> bool:
        """判断工具调用是否可进入只读并发批次。

        使用当前阶段已解析的工具与参数判断并行支持兼容度。

        Args:
            prepared (PreparedToolCall): 已完成 lookup、校验与鉴权的调用。

        Returns:
            bool: 回报其支持策略状态。分类异常或无效调用保守退回串行。
        """
        if (
            prepared.tool is None
            or prepared.preflight_result is not None
            or prepared.human_request is not None
        ):
            return False
        try:
            return prepared.tool.is_read_only(
                prepared.arguments
            ) and prepared.tool.is_concurrency_safe(prepared.arguments)
        except Exception:
            return False

    def _record_breaker_result(self, tool_name: str, result: ToolResult) -> None:
        """将执行结果写入熔断器。"""
        if self.circuit_breaker is not None:
            self.circuit_breaker.after_result(tool_name, result)

    # endregion


def _tool_error_code_and_message(
    message: str,
    *,
    allow_structured: bool,
) -> tuple[str, str]:
    """从工具异常消息中提取稳定错误码。

    解析系统抛出的不规则报错短语，将其按系统规格分离为大写键和文字段的结构。

    Args:
        message (str): 捕获得到的待分析文本。
        allow_structured (bool): 是否允许对其执行结构化键值分析，否则直接按明文输出。

    Returns:
        tuple[str, str]: 可直接作为字典键值存储的安全双元组。
    """
    if not allow_structured:
        return "EXECUTION_ERROR", message
    match = re.match(r"^([A-Z][A-Z0-9_]+):\s*(.*)$", message)
    if match is None:
        return "EXECUTION_ERROR", message
    return match.group(1), match.group(2)


def _copy_context_for_parallel_call(
    context: ToolExecutionContext,
) -> ToolExecutionContext:
    """新低层调用获得独立命令槽；共享读取状态与取消，不借用 Runtime owner。"""
    return context.model_copy(
        update={
            "metadata": deepcopy(context.metadata),
            "command_stop_slot": CommandStopSlot(),
            "execution_control": None,
        }
    )


def _human_interaction_request(
    tool_use: ToolUseBlock,
    tool: BaseTool,
    validated_input: BaseModel | dict[str, Any],
    arguments: dict[str, Any],
    decision: PermissionDecision,
    context: ToolExecutionContext,
) -> HumanInteractionRequest | None:
    prompt: PermissionPrompt | QuestionPrompt | None = None
    if tool.definition.group == "human":
        builder = getattr(tool, "build_interaction_prompt", None)
        if callable(builder):
            prompt = builder(params=validated_input)
    elif decision.effect is PermissionEffect.REQUIRE_HUMAN:
        prompt = PermissionPrompt(reason=decision.reason)
    if prompt is None:
        return None

    tool_call = _tool_call_snapshot(tool_use, tool, arguments, context)
    return HumanInteractionRequest(tool_call=tool_call, prompt=prompt)


def _tool_call_snapshot(
    tool_use: ToolUseBlock,
    tool: BaseTool,
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolCallSnapshot:
    run_id = str(context.metadata.get("run_id", ""))
    workspace_root = str(context.workspace_root.resolve())
    return ToolCallSnapshot(
        tool_call_id=tool_use.id,
        tool_name=tool.name,
        arguments=params,
        workspace_root=workspace_root,
        fingerprint=make_call_fingerprint(
            session_id=context.session_id,
            run_id=run_id,
            tool_call_id=tool_use.id,
            tool_name=tool.name,
            arguments=params,
            workspace_root=workspace_root,
        ),
    )
