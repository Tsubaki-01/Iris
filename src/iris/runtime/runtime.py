"""单轮 Agent runtime。

本模块从 session history、context 和当前用户输入构造一次 `LLMRequest`，
调用注入的 provider，并把 assistant 回复写回 session。
memory 只支持显式 opt-in 注入；工具执行只做一次 bridge，bounded loop 留给后续阶段组合。

Example:
    runtime = AgentRuntime(
        agent_config=config,
        context_input=context_input,
        provider=fake_provider,
    )
    result = await runtime.run_turn("你好")
"""

# region imports
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..agents import AgentConfig
from ..context import ContextBuilder, ContextBuildInput
from ..exceptions import (
    HITLCheckpointInvalidError,
    HITLExecutionOutcomeUnknownError,
    HITLResponseRequiredError,
)
from ..hitl import (
    HumanInteraction,
    HumanInteractionService,
    InMemoryInteractionStore,
    InteractionResumePhase,
    InteractionStatus,
    InteractionStore,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)
from ..message import LLMRequest, LLMResponse, Msg, ToolUseBlock
from ..session import InMemorySessionStore, SessionStore
from ..tools import (
    DefaultPermissionPolicy,
    PermissionPolicy,
    PreparedToolCall,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryView,
    ToolResult,
)
from .assembler import RuntimeMessageAssembler
from .checkpoint import build_hitl_checkpoint, validate_hitl_checkpoint
from .errors import error_result, normalize_runtime_error, tool_error_info
from .memory_context import prepare_memory_context_input
from .metadata import build_run_metadata, synchronize_resume_metadata
from .models import (
    RuntimeContinuationClaim,
    RuntimeErrorInfo,
    RuntimeOptions,
    RuntimeStatus,
    RuntimeTurnResult,
    ToolBridgeResult,
    ToolErrorPolicy,
)
from .resume import (
    append_resumed_result,
    commit_ready_interaction,
    load_resumable_interaction,
    resolve_interaction_result,
)
from .tool_bridge import ToolBridge

# endregion

if TYPE_CHECKING:
    from ..memory import MemoryContextBuilder, MemoryService


class RuntimeProvider(Protocol):
    """Runtime 调用的 provider 最小协议。

    运行时只依赖 provider-neutral 的 `LLMRequest` / `LLMResponse`，这样测试可注入
    FakeProvider，生产路径也可复用真实 provider client，而不让 runtime 读取厂商
    raw payload。

    Example:
        response = await provider.complete(request)
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """执行一次非流式 LLM 请求。

        Args:
            request (LLMRequest): Runtime 已组装完成的 provider-neutral 请求。

        Returns:
            LLMResponse: Provider 适配层归一化后的响应。
        """


class AgentRuntime:
    """编排一次本地 Agent 单轮调用。

    `AgentRuntime` 只负责调用顺序和边界错误归一化；context 构建、消息装配、
    provider 调用和 session 存储仍由各自组件承担。这样单轮 runtime 可先稳定下来，
    后续 memory、工具桥和 loop 可以在同一边界上继续组合。

    Example:
        runtime = AgentRuntime(
            agent_config=config,
            context_input=context_input,
            provider=fake_provider,
        )
        result = await runtime.run_turn("当前问题")
    """

    def __init__(
        self,
        *,
        agent_config: AgentConfig,
        context_input: ContextBuildInput,
        provider: RuntimeProvider,
        session_store: SessionStore | None = None,
        context_builder: ContextBuilder | None = None,
        assembler: RuntimeMessageAssembler | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_view: ToolRegistryView | None = None,
        tool_executor: ToolExecutor | None = None,
        workspace_root: Path | None = None,
        permission_policy: PermissionPolicy | None = None,
        interaction_store: InteractionStore | None = None,
        interaction_service: HumanInteractionService | None = None,
        memory_service: MemoryService | None = None,
        memory_context_builder: MemoryContextBuilder | None = None,
    ) -> None:
        """创建单轮 runtime。

        Args:
            agent_config (AgentConfig): 已加载并通过配置校验的 Agent 配置。
            context_input (ContextBuildInput): ContextBuilder 的输入数据。
            provider (RuntimeProvider): 本轮调用使用的 provider 实现。
            session_store (SessionStore | None): 可选会话存储；默认使用内存 store。
            context_builder (ContextBuilder | None): 可选 context 构建器，便于测试注入。
            assembler (RuntimeMessageAssembler | None): 可选消息装配器，便于测试注入。
            tool_registry (ToolRegistry | None): 可选工具注册表，供后续工具桥阶段复用。
            tool_view (ToolRegistryView | None): 可选工具视图，默认由注册表创建。
            tool_executor (ToolExecutor | None): 可选工具执行器，默认使用同一注册表和权限策略。
            workspace_root (Path | None): 工具执行时使用的 workspace 根路径。
            permission_policy (PermissionPolicy | None): 工具权限策略。
            interaction_store (InteractionStore | None): 人工交互的持久化存储。
            interaction_service (HumanInteractionService | None): 人工交互生命周期服务。
            memory_service (MemoryService | None): 显式 memory 阶段复用的可选服务。
            memory_context_builder (MemoryContextBuilder | None): 显式 memory 结果裁剪器。
        """
        from ..memory import MemoryContextBuilder

        self.agent_config = agent_config
        self.context_input = context_input
        self.provider = provider
        self.session_store = session_store or InMemorySessionStore()
        self.context_builder = context_builder or ContextBuilder()
        self.assembler = assembler or RuntimeMessageAssembler()
        base_registry = tool_registry
        if base_registry is None and tool_view is not None:
            base_registry = tool_view.registry
        self.tool_registry = base_registry or ToolRegistry()
        self.tool_view = tool_view or self.tool_registry.view()
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.tool_executor = tool_executor or ToolExecutor(
            self.tool_registry,
            permission_policy=self.permission_policy,
        )
        self.tool_bridge = ToolBridge(
            tool_view=self.tool_view,
            tool_executor=self.tool_executor,
        )
        self.interaction_store = interaction_store or InMemoryInteractionStore()
        self.interaction_service = interaction_service or HumanInteractionService(
            self.interaction_store
        )
        self.memory_service = memory_service
        self.memory_context_builder = memory_context_builder or MemoryContextBuilder()

    def _create_waiting_interaction(
        self,
        *,
        run_mode: str,
        assistant_message: Msg,
        response: LLMResponse,
        runtime_options: RuntimeOptions,
        metadata: Mapping[str, Any],
        step_index: int,
        all_tool_results: list[ToolResult],
    ) -> HumanInteraction | None:
        """预检工具批次，并在第一处人工 gate 创建持久化 interaction。"""
        plan = self.tool_bridge.preflight_once(
            assistant_message=assistant_message,
            session_id=runtime_options.session_id,
            run_id=runtime_options.run_id,
            agent_id=self.agent_config.name,
            workspace_root=self.workspace_root,
            permission_mode=self.agent_config.permissions.writes,
            metadata=metadata,
            tools_enabled=runtime_options.include_tools,
        )
        gate = plan.first_human_gate
        if gate is None or gate.human_request is None:
            return None

        checkpoint = build_hitl_checkpoint(
            run_mode=run_mode,
            agent_name=self.agent_config.name,
            runtime_options=runtime_options,
            assistant_message=assistant_message,
            response=response,
            step_index=step_index,
            next_tool_index=0,
            batch_results=[],
            all_tool_results=all_tool_results,
            read_state=self.tool_bridge.read_state(runtime_options.session_id),
        )
        return self._create_interaction(
            prepared=gate,
            checkpoint=checkpoint,
            session_id=runtime_options.session_id,
            run_id=runtime_options.run_id,
            step_index=step_index,
        )

    def _create_interaction(
        self,
        *,
        prepared: PreparedToolCall,
        checkpoint: dict[str, Any],
        session_id: str,
        run_id: str,
        step_index: int,
    ) -> HumanInteraction:
        """通过统一 service 路径持久化一处人工 gate。"""
        if prepared.human_request is None:
            raise HITLCheckpointInvalidError("预检工具调用缺少 HITL interaction 请求")
        return self.interaction_service.create(
            prepared.human_request,
            session_id=session_id,
            run_id=run_id,
            step_index=step_index,
            checkpoint=checkpoint,
        )

    def load_resumable_interaction(self, session_id: str) -> HumanInteraction | None:
        """读取当前 session 可安全交给 ``resume()`` 的 interaction。

        latest run marker 是恢复发现的主索引；唯一 pending interaction 只用于覆盖新 gate
        已创建但 marker 尚未写入的窄窗口。该查询不会修改 interaction 或 session。

        Args:
            session_id: 要检查的 session 标识。

        Returns:
            可恢复的 interaction；当前没有恢复目标时返回 ``None``。

        Raises:
            HITLCheckpointInvalidError: marker 缺失目标、跨 session 或与 pending 状态冲突。
        """
        return load_resumable_interaction(
            session_store=self.session_store,
            interaction_service=self.interaction_service,
            session_id=session_id,
        )

    async def resume(
        self,
        interaction_id: str,
        response: PermissionInteractionResponse | QuestionInteractionResponse | None = None,
    ) -> RuntimeTurnResult:
        """恢复已等待人工响应的一次工具调用。"""
        try:
            interaction = self.interaction_service.get(interaction_id)
            if interaction.status is InteractionStatus.PENDING:
                if response is None:
                    raise HITLResponseRequiredError("HITL interaction 尚未收到 response")
                interaction = self.interaction_service.resolve(interaction_id, response)
            elif interaction.status is InteractionStatus.RESOLVED and response is not None:
                interaction = self.interaction_service.resolve(interaction_id, response)
            checkpoint, options = validate_hitl_checkpoint(
                interaction,
                agent_name=self.agent_config.name,
            )
            self.tool_bridge.restore_read_state(
                interaction.session_id,
                (
                    checkpoint.get("read_state")
                    if isinstance(checkpoint.get("read_state"), dict)
                    else None
                ),
            )
            calls = [ToolUseBlock.model_validate(item) for item in checkpoint["tool_calls"]]
            plan = self.tool_bridge.preflight_once(
                assistant_message=Msg.assistant(calls),
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                agent_id=self.agent_config.name,
                workspace_root=self.workspace_root,
                permission_mode=self.agent_config.permissions.writes,
                metadata=options.metadata,
                tools_enabled=options.include_tools,
            )
            prepared = next(
                (
                    item
                    for item in plan.calls
                    if item.tool_use.id == interaction.request.tool_call.tool_call_id
                ),
                None,
            )
            if prepared is None or prepared.human_request is None:
                raise HITLCheckpointInvalidError("HITL checkpoint 工具调用不再需要人工确认")
            if interaction.request != prepared.human_request:
                raise HITLCheckpointInvalidError("HITL interaction 请求与当前工具定义不匹配")
            if interaction.status is InteractionStatus.CONSUMED:
                if interaction.resume_phase is InteractionResumePhase.CLAIMED:
                    raise HITLExecutionOutcomeUnknownError("HITL 工具执行结果未知，拒绝重放")
                if checkpoint.get("continuation_claim") is not None:
                    raise HITLExecutionOutcomeUnknownError(
                        "HITL continuation 执行结果未知，拒绝重放"
                    )
                committed = await commit_ready_interaction(
                    interaction=interaction,
                    session_store=self.session_store,
                    interaction_service=self.interaction_service,
                    agent_id=self.agent_config.name,
                )
                interaction = self.interaction_service.get(interaction_id)
                if checkpoint.get("continuation_complete") is True:
                    return synchronize_resume_metadata(
                        session_store=self.session_store,
                        result=committed,
                    )
                return synchronize_resume_metadata(
                    session_store=self.session_store,
                    result=await self._resume_batch(
                        committed=committed,
                        interaction=interaction,
                        checkpoint=checkpoint,
                        options=options,
                        plan=plan,
                        next_index=int(checkpoint["next_tool_index"]),
                    ),
                )

            interaction = self.interaction_service.claim(interaction_id, checkpoint)
            current_index = plan.calls.index(prepared)
            next_index = int(checkpoint["next_tool_index"])
            if next_index > current_index:
                raise HITLCheckpointInvalidError("HITL checkpoint 工具游标越过当前 gate")
            prefix_results = await self._execute_resumed_range(
                plan=plan,
                start_index=next_index,
                end_index=current_index,
                options=options,
                interaction=interaction,
            )
            result = await resolve_interaction_result(
                interaction=interaction,
                prepared=prepared,
                tool_executor=self.tool_executor,
                tool_context=self._tool_context(options),
            )
            ready_checkpoint = dict(checkpoint)
            batch_results = [*prefix_results, result]
            all_tool_results = [
                *[ToolResult.model_validate(item) for item in checkpoint["all_tool_results"]],
                *batch_results,
            ]
            ready_checkpoint.update(
                pending_result=result.model_dump(mode="json"),
                next_tool_index=current_index + 1,
                batch_results=[item.model_dump(mode="json") for item in batch_results],
                all_tool_results=[item.model_dump(mode="json") for item in all_tool_results],
            )
            interaction = self.interaction_service.update_consumed(
                interaction_id,
                InteractionResumePhase.RESULT_READY,
                ready_checkpoint,
                expected_phase=interaction.resume_phase,
                expected_version=interaction.version,
            )
            committed = await commit_ready_interaction(
                interaction=interaction,
                session_store=self.session_store,
                interaction_service=self.interaction_service,
                agent_id=self.agent_config.name,
            )
            interaction = self.interaction_service.get(interaction_id)
            return synchronize_resume_metadata(
                session_store=self.session_store,
                result=await self._resume_batch(
                    committed=committed,
                    interaction=interaction,
                    checkpoint=ready_checkpoint,
                    options=options,
                    plan=plan,
                    next_index=current_index + 1,
                ),
            )
        except Exception as exc:
            loaded_interaction = self.interaction_store.load_interaction(interaction_id)
            failed_result = error_result(
                session_id=(
                    loaded_interaction.session_id if loaded_interaction is not None else "default"
                ),
                run_id=loaded_interaction.run_id if loaded_interaction is not None else "resume",
                error=normalize_runtime_error(exc),
            )
            if loaded_interaction is None:
                return failed_result
            return synchronize_resume_metadata(
                session_store=self.session_store,
                result=failed_result,
            )

    async def _resume_batch(
        self,
        *,
        committed: RuntimeTurnResult,
        interaction: HumanInteraction,
        checkpoint: dict[str, Any],
        options: RuntimeOptions,
        plan: Any,
        next_index: int,
    ) -> RuntimeTurnResult:
        """从下一条未完成调用继续同批工具，直至下一个 gate 或批次结束。"""
        results = list(committed.tool_results)
        messages = list(committed.tool_result_messages)
        batch_results = [ToolResult.model_validate(item) for item in checkpoint["batch_results"]]
        for index, prepared in enumerate(plan.calls[next_index:], start=next_index):
            if prepared.human_request is not None:
                interaction, checkpoint = self._update_resume_checkpoint(
                    interaction=interaction,
                    checkpoint=checkpoint,
                    next_tool_index=index,
                    batch_results=batch_results,
                    all_tool_results=results,
                    continuation_complete=True,
                )
                pending = self._create_followup_interaction(
                    interaction=interaction,
                    checkpoint=checkpoint,
                    prepared=prepared,
                    next_index=index,
                    results=results,
                )
                return committed.model_copy(
                    update={
                        "status": RuntimeStatus.WAITING_HUMAN,
                        "pending_interaction": pending,
                        "tool_results": results,
                        "tool_result_messages": messages,
                    }
                )
            interaction, checkpoint = self._claim_resumed_continuation(
                interaction=interaction,
                checkpoint=checkpoint,
                claim=RuntimeContinuationClaim(
                    kind="tool",
                    next_tool_index=index,
                    tool_call_id=prepared.tool_use.id,
                ),
            )
            result = await self.tool_executor.execute_prepared(
                prepared, self._tool_context(options)
            )
            message = append_resumed_result(
                result=result,
                session_store=self.session_store,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
                agent_id=self.agent_config.name,
            )
            results.append(result)
            messages.append(message)
            batch_results.append(result)
            interaction, checkpoint = self._complete_resumed_continuation(
                interaction=interaction,
                checkpoint=checkpoint,
                next_tool_index=index + 1,
                batch_results=batch_results,
                all_tool_results=results,
            )
        completed = committed.model_copy(
            update={"tool_results": results, "tool_result_messages": messages}
        )
        if checkpoint.get("run_mode") == "loop":
            interaction, checkpoint = self._claim_resumed_continuation(
                interaction=interaction,
                checkpoint=checkpoint,
                claim=RuntimeContinuationClaim(
                    kind="loop",
                    next_tool_index=len(plan.calls),
                ),
            )
            completed = await self._continue_resumed_loop(completed, options)
            interaction, checkpoint = self._complete_resumed_continuation(
                interaction=interaction,
                checkpoint=checkpoint,
                next_tool_index=len(plan.calls),
                batch_results=batch_results,
                all_tool_results=completed.tool_results,
                continuation_complete=True,
            )
        else:
            interaction, checkpoint = self._update_resume_checkpoint(
                interaction=interaction,
                checkpoint=checkpoint,
                next_tool_index=len(plan.calls),
                batch_results=batch_results,
                all_tool_results=results,
                continuation_complete=True,
            )
        return completed

    async def _execute_resumed_range(
        self,
        *,
        plan: Any,
        start_index: int,
        end_index: int,
        options: RuntimeOptions,
        interaction: HumanInteraction,
    ) -> list[ToolResult]:
        """执行当前 gate 之前尚未完成的工具调用。"""
        results: list[ToolResult] = []
        for prepared in plan.calls[start_index:end_index]:
            if prepared.human_request is not None:
                raise HITLCheckpointInvalidError("HITL checkpoint 跳过了未处理的人工 gate")
            result = prepared.preflight_result
            if result is None:
                result = await self.tool_executor.execute_prepared(
                    prepared,
                    self._tool_context(options),
                )
            append_resumed_result(
                result=result,
                session_store=self.session_store,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
                agent_id=self.agent_config.name,
            )
            results.append(result)
        return results

    def _update_resume_checkpoint(
        self,
        *,
        interaction: HumanInteraction,
        checkpoint: dict[str, Any],
        next_tool_index: int,
        batch_results: list[ToolResult],
        all_tool_results: list[ToolResult],
        continuation_complete: bool = False,
    ) -> tuple[HumanInteraction, dict[str, Any]]:
        """保存当前批次的安全恢复游标。"""
        updated = dict(checkpoint)
        updated.update(
            next_tool_index=next_tool_index,
            batch_results=[item.model_dump(mode="json") for item in batch_results],
            all_tool_results=[item.model_dump(mode="json") for item in all_tool_results],
            read_state=(
                state.model_dump(mode="json")
                if (state := self.tool_bridge.read_state(interaction.session_id)) is not None
                else None
            ),
            continuation_complete=continuation_complete,
        )
        interaction = self.interaction_service.update_consumed(
            interaction.interaction_id,
            InteractionResumePhase.RESULT_COMMITTED,
            updated,
            expected_phase=interaction.resume_phase,
            expected_version=interaction.version,
        )
        return interaction, updated

    def _claim_resumed_continuation(
        self,
        *,
        interaction: HumanInteraction,
        checkpoint: dict[str, Any],
        claim: RuntimeContinuationClaim,
    ) -> tuple[HumanInteraction, dict[str, Any]]:
        """在恢复后的副作用执行前持久化 fail-closed claim。"""
        if (
            interaction.status is not InteractionStatus.CONSUMED
            or interaction.resume_phase is not InteractionResumePhase.RESULT_COMMITTED
        ):
            raise HITLCheckpointInvalidError("HITL continuation 只能从 result_committed 执行")
        if checkpoint.get("continuation_claim") is not None:
            raise HITLExecutionOutcomeUnknownError("HITL continuation 执行结果未知，拒绝重放")
        updated = dict(checkpoint)
        updated["continuation_claim"] = claim.model_dump(mode="json")
        interaction = self.interaction_service.update_consumed(
            interaction.interaction_id,
            InteractionResumePhase.RESULT_COMMITTED,
            updated,
            expected_phase=interaction.resume_phase,
            expected_version=interaction.version,
        )
        return interaction, updated

    def _complete_resumed_continuation(
        self,
        *,
        interaction: HumanInteraction,
        checkpoint: dict[str, Any],
        next_tool_index: int,
        batch_results: list[ToolResult],
        all_tool_results: list[ToolResult],
        continuation_complete: bool = False,
    ) -> tuple[HumanInteraction, dict[str, Any]]:
        """提交 continuation 结果、推进游标并原子清除 claim。"""
        if checkpoint.get("continuation_claim") is None:
            raise HITLCheckpointInvalidError("HITL continuation 缺少执行 claim")
        updated = dict(checkpoint)
        updated.update(
            next_tool_index=next_tool_index,
            batch_results=[item.model_dump(mode="json") for item in batch_results],
            all_tool_results=[item.model_dump(mode="json") for item in all_tool_results],
            read_state=(
                state.model_dump(mode="json")
                if (state := self.tool_bridge.read_state(interaction.session_id)) is not None
                else None
            ),
            continuation_claim=None,
            continuation_complete=continuation_complete,
        )
        interaction = self.interaction_service.update_consumed(
            interaction.interaction_id,
            InteractionResumePhase.RESULT_COMMITTED,
            updated,
            expected_phase=interaction.resume_phase,
            expected_version=interaction.version,
        )
        return interaction, updated

    def _create_followup_interaction(
        self,
        *,
        interaction: HumanInteraction,
        checkpoint: dict[str, Any],
        prepared: PreparedToolCall,
        next_index: int,
        results: list[ToolResult],
    ) -> HumanInteraction:
        """为批次中下一处人工 gate 创建独立 interaction。"""
        read_state = self.tool_bridge.read_state(interaction.session_id)
        next_checkpoint = dict(checkpoint)
        next_checkpoint.update(
            next_tool_index=next_index,
            batch_results=[result.model_dump(mode="json") for result in results],
            all_tool_results=[result.model_dump(mode="json") for result in results],
            pending_result=None,
            continuation_complete=False,
            read_state=read_state.model_dump(mode="json") if read_state is not None else None,
        )
        return self._create_interaction(
            prepared=prepared,
            checkpoint=next_checkpoint,
            session_id=interaction.session_id,
            run_id=interaction.run_id,
            step_index=interaction.step_index,
        )

    def _tool_context(self, options: RuntimeOptions) -> ToolExecutionContext:
        """从 checkpoint 还原执行工具所需的最小上下文。"""
        return ToolExecutionContext(
            workspace_root=self.workspace_root,
            session_id=options.session_id,
            agent_id=self.agent_config.name,
            permission_mode=self.agent_config.permissions.writes,
            metadata={**options.metadata, "run_id": options.run_id},
            read_state=self.tool_bridge.read_state(options.session_id),
        )

    async def _continue_resumed_loop(
        self,
        committed: RuntimeTurnResult,
        options: RuntimeOptions,
    ) -> RuntimeTurnResult:
        """将已提交的工具结果回灌 provider，继续被暂停的 loop。"""
        try:
            history = _load_history(self.session_store, committed.session_id)
            context_output = self.context_builder.build(
                prepare_memory_context_input(
                    self.context_input,
                    options=options,
                    memory_service=self.memory_service,
                    memory_context_builder=self.memory_context_builder,
                )
            )
            request = self.assembler.build_request(
                agent_config=self.agent_config,
                context_output=context_output,
                history=history,
                current_input=None,
            )
            request = _apply_tool_schemas(
                _apply_request_options(request, options.request_options),
                include_tools=options.include_tools,
                tool_view=self.tool_view,
                provider=self.agent_config.model.provider,
            )
            response = await self.provider.complete(request)
            assistant = response.to_msg()
            history.append(assistant)
            self.session_store.save_messages(
                committed.session_id, [item.model_dump(mode="json") for item in history]
            )
            if not assistant.has_tool_calls:
                return committed.model_copy(
                    update={"assistant_message": assistant, "steps": committed.steps + 1}
                )
            pending = self._create_waiting_interaction(
                run_mode="loop",
                assistant_message=assistant,
                response=response,
                runtime_options=options,
                metadata=options.metadata,
                step_index=committed.steps,
                all_tool_results=committed.tool_results,
            )
            if pending is not None:
                return committed.model_copy(
                    update={
                        "status": RuntimeStatus.WAITING_HUMAN,
                        "assistant_message": assistant,
                        "steps": committed.steps + 1,
                        "pending_interaction": pending,
                    }
                )
            bridge = await self.tool_bridge.execute_once(
                assistant_message=assistant,
                session_id=committed.session_id,
                run_id=committed.run_id,
                step_index=committed.steps,
                agent_id=self.agent_config.name,
                workspace_root=self.workspace_root,
                permission_mode=self.agent_config.permissions.writes,
                session_store=self.session_store,
                metadata=options.metadata,
                tools_enabled=options.include_tools,
            )
            if bridge.messages:
                history.extend(bridge.messages)
                self.session_store.save_messages(
                    committed.session_id,
                    [item.model_dump(mode="json") for item in history],
                )
            continued = committed.model_copy(
                update={
                    "assistant_message": assistant,
                    "tool_results": [*committed.tool_results, *bridge.results],
                    "tool_result_messages": [*committed.tool_result_messages, *bridge.messages],
                    "steps": committed.steps + 1,
                }
            )
            if _should_stop_on_tool_error(options, bridge):
                return RuntimeTurnResult(
                    session_id=continued.session_id,
                    run_id=continued.run_id,
                    status=RuntimeStatus.ERROR,
                    assistant_message=assistant,
                    tool_results=continued.tool_results,
                    tool_result_messages=continued.tool_result_messages,
                    steps=continued.steps,
                    error=tool_error_info(bridge),
                    metadata=continued.metadata,
                )
            if continued.steps >= options.loop.max_steps:
                return RuntimeTurnResult(
                    session_id=continued.session_id,
                    run_id=continued.run_id,
                    status=RuntimeStatus.MAX_STEPS,
                    assistant_message=assistant,
                    tool_results=continued.tool_results,
                    tool_result_messages=continued.tool_result_messages,
                    steps=continued.steps,
                    error=RuntimeErrorInfo(
                        code="MAX_STEPS_REACHED",
                        message=f"已达到最大 loop 步数: {options.loop.max_steps}",
                        source="runtime",
                    ),
                )
            return await self._continue_resumed_loop(continued, options)
        except Exception as exc:
            return error_result(
                session_id=committed.session_id,
                run_id=committed.run_id,
                error=normalize_runtime_error(exc),
                steps=committed.steps,
            )

    async def run_turn(
        self,
        user_input: str,
        *,
        options: RuntimeOptions | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTurnResult:
        """执行一次 provider call 并保存用户输入与 assistant 回复。

        Args:
            user_input (str): 当前用户输入内容。
            options (RuntimeOptions | None): 本轮 session、run id 和 request 覆盖选项。
            metadata (Mapping[str, Any] | None): 调用方传入的追踪信息，只进入 run metadata。

        Returns:
            RuntimeTurnResult: 成功时包含 assistant 消息；失败时包含归一化错误。
        """
        runtime_options = options or RuntimeOptions()
        run_metadata = {**runtime_options.metadata, **dict(metadata or {})}
        session_id = runtime_options.session_id
        run_id = runtime_options.run_id

        # --- 1. Build request ---
        try:
            history = _load_history(self.session_store, session_id)
            context_input = prepare_memory_context_input(
                self.context_input,
                options=runtime_options,
                memory_service=self.memory_service,
                memory_context_builder=self.memory_context_builder,
            )
            context_output = self.context_builder.build(context_input)
            current_input = Msg.user(user_input)
            turn_messages = self.assembler.build_turn_messages(
                context_output=context_output,
                current_input=current_input,
            )
            request = self.assembler.build_request(
                agent_config=self.agent_config,
                context_output=context_output,
                history=history,
                current_input=current_input,
            )
            request = _apply_request_options(request, runtime_options.request_options)
            request = _apply_tool_schemas(
                request,
                include_tools=runtime_options.include_tools,
                tool_view=self.tool_view,
                provider=self.agent_config.model.provider,
            )
        except Exception as exc:
            return error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                metadata=run_metadata,
            )

        # --- 2. Call provider ---
        try:
            response = await self.provider.complete(request)
            assistant_message = response.to_msg()
        except Exception as exc:
            return error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                metadata=run_metadata,
            )

        # --- 3. Persist result ---
        messages = [*history, *turn_messages, assistant_message]
        try:
            self.session_store.save_messages(
                session_id,
                [message.model_dump(mode="json") for message in messages],
            )
            pending_interaction = self._create_waiting_interaction(
                run_mode="turn",
                assistant_message=assistant_message,
                response=response,
                runtime_options=runtime_options,
                metadata=run_metadata,
                step_index=0,
                all_tool_results=[],
            )
            if pending_interaction is not None:
                self.session_store.save_run_metadata(
                    session_id,
                    build_run_metadata(
                        existing=self.session_store.load_run_metadata(session_id),
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.WAITING_HUMAN,
                        provider=self.agent_config.model.provider,
                        response=response,
                        message_count=len(messages),
                        metadata=run_metadata,
                        waiting_human=True,
                        interaction_id=pending_interaction.interaction_id,
                    ),
                )
                return RuntimeTurnResult(
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.WAITING_HUMAN,
                    assistant_message=assistant_message,
                    steps=1,
                    pending_interaction=pending_interaction,
                    metadata=run_metadata,
                )
            bridge_result = await self.tool_bridge.execute_once(
                assistant_message=assistant_message,
                session_id=session_id,
                run_id=run_id,
                step_index=0,
                agent_id=self.agent_config.name,
                workspace_root=self.workspace_root,
                permission_mode=self.agent_config.permissions.writes,
                session_store=self.session_store,
                metadata=run_metadata,
                tools_enabled=runtime_options.include_tools,
            )
            if bridge_result.messages:
                messages.extend(bridge_result.messages)
                self.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )
            self.session_store.save_run_metadata(
                session_id,
                build_run_metadata(
                    existing=self.session_store.load_run_metadata(session_id),
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.OK,
                    provider=self.agent_config.model.provider,
                    response=response,
                    message_count=len(messages),
                    metadata=run_metadata,
                    tool_count=len(bridge_result.results),
                ),
            )
        except Exception as exc:
            error = normalize_runtime_error(exc)
            try:
                self.session_store.save_run_metadata(
                    session_id,
                    build_run_metadata(
                        existing=self.session_store.load_run_metadata(session_id),
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.ERROR,
                        provider=self.agent_config.model.provider,
                        response=response,
                        message_count=len(messages),
                        metadata=run_metadata,
                        error=error,
                    ),
                )
            except Exception:
                pass
            return error_result(
                session_id=session_id,
                run_id=run_id,
                error=error,
                assistant_message=assistant_message,
                metadata=run_metadata,
            )

        return RuntimeTurnResult(
            session_id=session_id,
            run_id=run_id,
            status=RuntimeStatus.OK,
            assistant_message=assistant_message,
            tool_result_messages=bridge_result.messages,
            tool_results=bridge_result.results,
            steps=1,
            metadata=run_metadata,
        )

    async def run_loop(
        self,
        user_input: str,
        *,
        options: RuntimeOptions | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTurnResult:
        """执行有界 tool loop。

        当前用户输入只进入第一步。后续步骤从 session history 重新装配请求，
        因此上一轮保存的 tool result message 会自然进入下一次 provider 请求。

        Args:
            user_input (str): 当前用户输入内容。
            options (RuntimeOptions | None): 本轮 session、run id、request 和 loop 选项。
            metadata (Mapping[str, Any] | None): 调用方传入的追踪信息。

        Returns:
            RuntimeTurnResult: loop 的最终助手消息、工具结果和状态。
        """
        runtime_options = options or RuntimeOptions()
        run_metadata = {**runtime_options.metadata, **dict(metadata or {})}
        session_id = runtime_options.session_id
        run_id = runtime_options.run_id
        max_steps = runtime_options.loop.max_steps
        all_tool_results: list[ToolResult] = []
        all_tool_messages: list[Msg] = []
        latest_assistant: Msg | None = None
        latest_response: LLMResponse | None = None

        for step_index in range(max_steps):
            step_number = step_index + 1
            current_input = Msg.user(user_input) if step_index == 0 else None
            try:
                history = _load_history(self.session_store, session_id)
                context_input = prepare_memory_context_input(
                    self.context_input,
                    options=runtime_options,
                    memory_service=self.memory_service,
                    memory_context_builder=self.memory_context_builder,
                )
                context_output = self.context_builder.build(context_input)
                turn_messages = self.assembler.build_turn_messages(
                    context_output=context_output,
                    current_input=current_input,
                )
                request = self.assembler.build_request(
                    agent_config=self.agent_config,
                    context_output=context_output,
                    history=history,
                    current_input=current_input,
                )
                request = _apply_request_options(
                    request,
                    runtime_options.request_options,
                )
                request = _apply_tool_schemas(
                    request,
                    include_tools=runtime_options.include_tools,
                    tool_view=self.tool_view,
                    provider=self.agent_config.model.provider,
                )
                latest_response = await self.provider.complete(request)
                latest_assistant = latest_response.to_msg()
            except Exception as exc:
                return error_result(
                    session_id=session_id,
                    run_id=run_id,
                    error=normalize_runtime_error(exc),
                    assistant_message=latest_assistant,
                    steps=step_number,
                    metadata=run_metadata,
                )

            messages = [*history, *turn_messages, latest_assistant]

            try:
                self.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )

                if not latest_assistant.has_tool_calls:
                    self.session_store.save_run_metadata(
                        session_id,
                        build_run_metadata(
                            existing=self.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.OK,
                            provider=self.agent_config.model.provider,
                            response=latest_response,
                            message_count=len(messages),
                            metadata=run_metadata,
                            steps=step_number,
                            tool_count=len(all_tool_results),
                        ),
                    )
                    return RuntimeTurnResult(
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.OK,
                        assistant_message=latest_assistant,
                        tool_result_messages=all_tool_messages,
                        tool_results=all_tool_results,
                        steps=step_number,
                        metadata=run_metadata,
                    )

                pending_interaction = self._create_waiting_interaction(
                    run_mode="loop",
                    assistant_message=latest_assistant,
                    response=latest_response,
                    runtime_options=runtime_options,
                    metadata=run_metadata,
                    step_index=step_index,
                    all_tool_results=all_tool_results,
                )
                if pending_interaction is not None:
                    self.session_store.save_run_metadata(
                        session_id,
                        build_run_metadata(
                            existing=self.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.WAITING_HUMAN,
                            provider=self.agent_config.model.provider,
                            response=latest_response,
                            message_count=len(messages),
                            metadata=run_metadata,
                            steps=step_number,
                            tool_count=len(all_tool_results),
                            waiting_human=True,
                            interaction_id=pending_interaction.interaction_id,
                        ),
                    )
                    return RuntimeTurnResult(
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.WAITING_HUMAN,
                        assistant_message=latest_assistant,
                        tool_result_messages=all_tool_messages,
                        tool_results=all_tool_results,
                        steps=step_number,
                        pending_interaction=pending_interaction,
                        metadata=run_metadata,
                    )

                bridge_result = await self.tool_bridge.execute_once(
                    assistant_message=latest_assistant,
                    session_id=session_id,
                    run_id=run_id,
                    step_index=step_index,
                    agent_id=self.agent_config.name,
                    workspace_root=self.workspace_root,
                    permission_mode=self.agent_config.permissions.writes,
                    session_store=self.session_store,
                    metadata=run_metadata,
                    tools_enabled=runtime_options.include_tools,
                )
                messages.extend(bridge_result.messages)
                self.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )
            except Exception as exc:
                return error_result(
                    session_id=session_id,
                    run_id=run_id,
                    error=normalize_runtime_error(exc),
                    assistant_message=latest_assistant,
                    steps=step_number,
                    metadata=run_metadata,
                )

            all_tool_results.extend(bridge_result.results)
            all_tool_messages.extend(bridge_result.messages)

            if _should_stop_on_tool_error(runtime_options, bridge_result):
                error = tool_error_info(bridge_result)
                try:
                    self.session_store.save_run_metadata(
                        session_id,
                        build_run_metadata(
                            existing=self.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.ERROR,
                            provider=self.agent_config.model.provider,
                            response=latest_response,
                            message_count=len(messages),
                            metadata=run_metadata,
                            steps=step_number,
                            tool_count=len(all_tool_results),
                            error=error,
                        ),
                    )
                except Exception as exc:
                    return error_result(
                        session_id=session_id,
                        run_id=run_id,
                        error=normalize_runtime_error(exc),
                        assistant_message=latest_assistant,
                        steps=step_number,
                        metadata=run_metadata,
                    )
                return RuntimeTurnResult(
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.ERROR,
                    assistant_message=latest_assistant,
                    tool_result_messages=all_tool_messages,
                    tool_results=all_tool_results,
                    steps=step_number,
                    error=error,
                    metadata=run_metadata,
                )

        error = RuntimeErrorInfo(
            code="MAX_STEPS_REACHED",
            message=f"已达到最大 loop 步数: {max_steps}",
            source="runtime",
            details={"max_steps": max_steps},
        )
        max_step_metadata = {**run_metadata, "max_steps": max_steps}
        try:
            self.session_store.save_run_metadata(
                session_id,
                build_run_metadata(
                    existing=self.session_store.load_run_metadata(session_id),
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.MAX_STEPS,
                    provider=self.agent_config.model.provider,
                    response=latest_response,
                    message_count=len(self.session_store.load_messages(session_id)),
                    metadata=max_step_metadata,
                    steps=max_steps,
                    tool_count=len(all_tool_results),
                    error=error,
                ),
            )
        except Exception as exc:
            return error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                assistant_message=latest_assistant,
                steps=max_steps,
                metadata=max_step_metadata,
            )
        return RuntimeTurnResult(
            session_id=session_id,
            run_id=run_id,
            status=RuntimeStatus.MAX_STEPS,
            assistant_message=latest_assistant,
            tool_result_messages=all_tool_messages,
            tool_results=all_tool_results,
            steps=max_steps,
            error=error,
            metadata=max_step_metadata,
        )


def _load_history(session_store: SessionStore, session_id: str) -> list[Msg]:
    """从 session 读取历史消息并恢复为 `Msg`。"""
    return [Msg.from_dict(message) for message in session_store.load_messages(session_id)]


def _apply_request_options(
    request: LLMRequest,
    request_options: Mapping[str, Any],
) -> LLMRequest:
    """应用本轮请求覆盖项。

    provider_options 字段采用合并策略，其余的进行补充或覆盖。
    """
    if not request_options:
        return request
    update = dict(request_options)
    if "provider_options" in update:
        provider_options = update["provider_options"]
        if isinstance(provider_options, Mapping):
            update["provider_options"] = {
                **request.provider_options,
                **dict(provider_options),
            }
    return request.model_copy(update=update)


def _apply_tool_schemas(
    request: LLMRequest,
    *,
    include_tools: bool,
    tool_view: ToolRegistryView,
    provider: str,
) -> LLMRequest:
    """按当前活动工具视图挂载 LiteLLM Chat 工具 schema。"""
    if not include_tools:
        return request
    tools = tool_view.active_schemas(
        provider="openai",
        api_style="chat",
    )
    return request.model_copy(update={"tools": tools})


def _should_stop_on_tool_error(
    options: RuntimeOptions,
    bridge_result: ToolBridgeResult,
) -> bool:
    """判断 loop 是否应在工具错误后停止。"""
    return options.loop.tool_error_policy == ToolErrorPolicy.STOP and any(
        result.is_error for result in bridge_result.results
    )


__all__ = ["AgentRuntime", "RuntimeProvider", "normalize_runtime_error"]
