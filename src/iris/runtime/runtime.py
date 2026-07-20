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
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..agents import AgentConfig
from ..context import ContextBuilder, ContextBuildInput
from ..exceptions import (
    HITLCheckpointInvalidError,
    HITLExecutionOutcomeUnknownError,
    HITLNotFoundError,
    HITLResponseRequiredError,
    IrisError,
)
from ..hitl import (
    HumanInteraction,
    HumanInteractionService,
    InMemoryInteractionStore,
    InteractionResumePhase,
    InteractionStatus,
    InteractionStore,
    PermissionInteractionRequest,
    PermissionInteractionResponse,
    QuestionInteractionRequest,
    QuestionInteractionResponse,
    make_call_fingerprint,
)
from ..message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock
from ..session import InMemorySessionStore, SessionStore
from ..tools import (
    DefaultPermissionPolicy,
    PermissionPolicy,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryView,
    ToolResult,
)
from .assembler import RuntimeMessageAssembler
from .memory import prepare_memory_context_input
from .models import (
    ProviderResponseSnapshot,
    RuntimeErrorInfo,
    RuntimeErrorSource,
    RuntimeHITLCheckpoint,
    RuntimeOptions,
    RuntimeOptionsSnapshot,
    RuntimeStatus,
    RuntimeTurnResult,
    ToolBridgeResult,
    ToolErrorPolicy,
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

        checkpoint = _build_hitl_checkpoint(
            run_mode=cast("str", run_mode),
            agent_name=self.agent_config.name,
            runtime_options=runtime_options,
            assistant_message=assistant_message,
            response=response,
            step_index=step_index,
            next_tool_index=0,
            batch_results=[],
            all_tool_results=all_tool_results,
            read_state=self.tool_bridge.read_state(runtime_options.session_id),
            call_fingerprint=_call_fingerprint(
                request=gate.human_request,
                session_id=runtime_options.session_id,
                run_id=runtime_options.run_id,
                tool_name=gate.tool_use.name,
                arguments=gate.validated_params,
                workspace_root=self.workspace_root,
            ),
        )
        request = gate.human_request
        if isinstance(request, PermissionInteractionRequest):
            return self.interaction_service.create_permission(
                session_id=runtime_options.session_id,
                run_id=runtime_options.run_id,
                step_index=step_index,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                reason=request.reason,
                workspace_root=request.workspace_root,
                checkpoint=checkpoint,
            )
        if isinstance(request, QuestionInteractionRequest):
            return self.interaction_service.create_question(
                session_id=runtime_options.session_id,
                run_id=runtime_options.run_id,
                step_index=step_index,
                tool_call_id=request.tool_call_id,
                question=request.question,
                options=request.options,
                checkpoint=checkpoint,
            )
        raise HITLCheckpointInvalidError("未知的 HITL interaction 请求")

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
        metadata = self.session_store.load_run_metadata(session_id)
        pending = self.interaction_service.list_pending(session_id)
        if len(pending) > 1:
            raise HITLCheckpointInvalidError(
                "同一 session 存在多个 pending HITL interaction",
                session_id=session_id,
            )
        fallback = pending[0] if pending else None
        latest = metadata.get("latest_run")
        if not isinstance(latest, dict) or latest.get("waiting_human") is not True:
            return fallback

        interaction_id = latest.get("interaction_id")
        if not isinstance(interaction_id, str) or not interaction_id.strip():
            if fallback is not None:
                return fallback
            raise HITLCheckpointInvalidError(
                "HITL waiting marker 缺少 interaction_id",
                session_id=session_id,
            )
        try:
            target = self.interaction_service.get(interaction_id)
        except HITLNotFoundError as exc:
            raise HITLCheckpointInvalidError(
                "HITL waiting marker 指向的 interaction 不存在",
                session_id=session_id,
                interaction_id=interaction_id,
            ) from exc
        if target.session_id != session_id:
            raise HITLCheckpointInvalidError(
                "HITL waiting marker 与请求 session 不匹配",
                session_id=session_id,
                interaction_id=interaction_id,
            )
        if fallback is None or fallback.interaction_id == target.interaction_id:
            return target
        if (
            target.status is InteractionStatus.CONSUMED
            and target.resume_phase is InteractionResumePhase.RESULT_COMMITTED
        ):
            return fallback
        raise HITLCheckpointInvalidError(
            "HITL waiting marker 与 pending interaction 冲突",
            session_id=session_id,
            interaction_id=interaction_id,
            pending_interaction_id=fallback.interaction_id,
        )

    def _synchronize_resume_metadata(self, result: RuntimeTurnResult) -> RuntimeTurnResult:
        """在向 host 返回 resume 结果前同步 latest run snapshot。"""
        try:
            existing = self.session_store.load_run_metadata(result.session_id)
            metadata = _build_resume_run_metadata(
                existing=existing,
                result=result,
                message_count=len(self.session_store.load_messages(result.session_id)),
            )
            self.session_store.save_run_metadata(result.session_id, metadata)
        except Exception as exc:
            return _error_result(
                session_id=result.session_id,
                run_id=result.run_id,
                error=normalize_runtime_error(exc),
                assistant_message=result.assistant_message,
                steps=result.steps,
                metadata=result.metadata,
            )
        return result

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
            checkpoint = interaction.checkpoint
            if checkpoint.get("checkpoint_version") != 1:
                raise HITLCheckpointInvalidError("不支持的 HITL checkpoint 版本")
            options = RuntimeOptions.model_validate(checkpoint["runtime_options"]["options"])
            if (
                checkpoint.get("agent_name") != self.agent_config.name
                or options.session_id != interaction.session_id
                or options.run_id != interaction.run_id
            ):
                raise HITLCheckpointInvalidError("HITL checkpoint 与当前 runtime 不匹配")
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
                (item for item in plan.calls if item.tool_use.id == interaction.tool_call_id), None
            )
            if prepared is None or prepared.human_request is None:
                raise HITLCheckpointInvalidError("HITL checkpoint 工具调用不再需要人工确认")
            if interaction.request != prepared.human_request:
                raise HITLCheckpointInvalidError("HITL interaction 请求与当前工具定义不匹配")
            expected_fingerprint = _call_fingerprint(
                request=prepared.human_request,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                tool_name=prepared.tool_use.name,
                arguments=prepared.validated_params,
                workspace_root=self.workspace_root,
            )
            if checkpoint.get("call_fingerprint") != expected_fingerprint:
                raise HITLCheckpointInvalidError("HITL checkpoint 调用指纹与当前 runtime 不匹配")
            if interaction.status is InteractionStatus.CONSUMED:
                if interaction.resume_phase is InteractionResumePhase.CLAIMED:
                    raise HITLExecutionOutcomeUnknownError("HITL 工具执行结果未知，拒绝重放")
                committed = await self._commit_ready_interaction(interaction)
                if checkpoint.get("continuation_complete") is True:
                    return self._synchronize_resume_metadata(committed)
                return self._synchronize_resume_metadata(
                    await self._resume_batch(
                        committed=committed,
                        interaction=interaction,
                        checkpoint=checkpoint,
                        options=options,
                        plan=plan,
                        next_index=int(checkpoint["next_tool_index"]),
                    )
                )

            self.interaction_service.claim(interaction_id, checkpoint)
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
            if isinstance(interaction.response, PermissionInteractionResponse):
                if interaction.response.decision == "approve":
                    result = await self.tool_executor.execute_prepared(
                        prepared,
                        self._tool_context(options),
                        approved_tool_call_id=interaction.tool_call_id,
                    )
                else:
                    result = ToolResult(
                        tool_use_id=interaction.tool_call_id,
                        tool_name=prepared.tool_use.name,
                        is_error=True,
                        error=ToolErrorInfo(code="USER_REJECTED", message="用户拒绝了工具调用"),
                    )
            else:
                if not isinstance(interaction.response, QuestionInteractionResponse):
                    raise HITLCheckpointInvalidError("HITL interaction 缺少有效 response")
                result = ToolResult(
                    tool_use_id=interaction.tool_call_id,
                    tool_name=prepared.tool_use.name,
                    content=[TextBlock(text=interaction.response.answer)],
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
            )
            committed = await self._commit_ready_interaction(interaction)
            return self._synchronize_resume_metadata(
                await self._resume_batch(
                    committed=committed,
                    interaction=interaction,
                    checkpoint=ready_checkpoint,
                    options=options,
                    plan=plan,
                    next_index=current_index + 1,
                )
            )
        except Exception as exc:
            loaded_interaction = self.interaction_store.load_interaction(interaction_id)
            error_result = _error_result(
                session_id=(
                    loaded_interaction.session_id if loaded_interaction is not None else "default"
                ),
                run_id=loaded_interaction.run_id if loaded_interaction is not None else "resume",
                error=normalize_runtime_error(exc),
            )
            if loaded_interaction is None:
                return error_result
            return self._synchronize_resume_metadata(error_result)

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
                checkpoint = self._update_resume_checkpoint(
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
            result = await self.tool_executor.execute_prepared(
                prepared, self._tool_context(options)
            )
            message = self._append_resumed_result(
                result=result,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
            )
            results.append(result)
            messages.append(message)
            batch_results.append(result)
            checkpoint = self._update_resume_checkpoint(
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
            completed = await self._continue_resumed_loop(completed, options)
        self._update_resume_checkpoint(
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
            self._append_resumed_result(
                result=result,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
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
    ) -> dict[str, Any]:
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
        self.interaction_service.update_consumed(
            interaction.interaction_id,
            InteractionResumePhase.RESULT_COMMITTED,
            updated,
        )
        return updated

    def _create_followup_interaction(
        self,
        *,
        interaction: HumanInteraction,
        checkpoint: dict[str, Any],
        prepared: Any,
        next_index: int,
        results: list[ToolResult],
    ) -> HumanInteraction:
        """为批次中下一处人工 gate 创建独立 interaction。"""
        request = prepared.human_request
        assert request is not None
        read_state = self.tool_bridge.read_state(interaction.session_id)
        next_checkpoint = dict(checkpoint)
        next_checkpoint.update(
            next_tool_index=next_index,
            batch_results=[result.model_dump(mode="json") for result in results],
            all_tool_results=[result.model_dump(mode="json") for result in results],
            pending_result=None,
            continuation_complete=False,
            read_state=read_state.model_dump(mode="json") if read_state is not None else None,
            call_fingerprint=_call_fingerprint(
                request=request,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                tool_name=prepared.tool_use.name,
                arguments=prepared.validated_params,
                workspace_root=self.workspace_root,
            ),
        )
        if isinstance(request, PermissionInteractionRequest):
            return self.interaction_service.create_permission(
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                reason=request.reason,
                workspace_root=request.workspace_root,
                checkpoint=next_checkpoint,
            )
        return self.interaction_service.create_question(
            session_id=interaction.session_id,
            run_id=interaction.run_id,
            step_index=interaction.step_index,
            tool_call_id=request.tool_call_id,
            question=request.question,
            options=request.options,
            checkpoint=next_checkpoint,
        )

    def _append_resumed_result(
        self, *, result: ToolResult, session_id: str, run_id: str, step_index: int
    ) -> Msg:
        """保存续跑工具结果，并使用稳定 event ID 避免重复追加。"""
        message = Msg.tool_result(
            tool_use_id=result.tool_use_id,
            content=result.model_content,
            is_error=result.is_error,
            name=result.tool_name,
            metadata=result.to_block_metadata(),
        )
        history = _load_history(self.session_store, session_id)
        history.append(message)
        self.session_store.save_messages(
            session_id, [item.model_dump(mode="json") for item in history]
        )
        self.session_store.append_tool_event(
            session_id,
            f"tool_result:{run_id}:{result.tool_use_id}",
            {
                "type": "tool_result",
                "tool_call_id": result.tool_use_id,
                "tool_name": result.tool_name,
                "status": "error" if result.is_error else "ok",
                "error": result.error.model_dump(mode="json") if result.error else None,
                "run_id": run_id,
                "step_index": step_index,
                "agent_id": self.agent_config.name,
                "metadata": {},
            },
        )
        return message

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
                    error=_tool_error_info(bridge),
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
            return _error_result(
                session_id=committed.session_id,
                run_id=committed.run_id,
                error=normalize_runtime_error(exc),
                steps=committed.steps,
            )

    async def _commit_ready_interaction(self, interaction: HumanInteraction) -> RuntimeTurnResult:
        """幂等写入已准备工具结果的消息与事件。"""
        checkpoint = interaction.checkpoint
        raw_result = checkpoint.get("pending_result")
        if not isinstance(raw_result, dict):
            raise HITLExecutionOutcomeUnknownError("HITL 已消费 interaction 缺少待提交结果")
        result = ToolResult.model_validate(raw_result)
        message = _tool_result_message(result)
        messages = _load_history(self.session_store, interaction.session_id)
        if not any(
            block.tool_use_id == result.tool_use_id
            for existing in messages
            for block in existing.tool_results
        ):
            messages.append(message)
            self.session_store.save_messages(
                interaction.session_id, [item.model_dump(mode="json") for item in messages]
            )
        event_id = f"tool_result:{interaction.run_id}:{interaction.tool_call_id}"
        self.session_store.append_tool_event(
            interaction.session_id,
            event_id,
            {
                "type": "tool_result",
                "tool_call_id": result.tool_use_id,
                "tool_name": result.tool_name,
                "status": "error" if result.is_error else "ok",
                "error": result.error.model_dump(mode="json") if result.error else None,
                "run_id": interaction.run_id,
                "step_index": interaction.step_index,
                "agent_id": self.agent_config.name,
                "metadata": {},
            },
        )
        if interaction.resume_phase is not InteractionResumePhase.RESULT_COMMITTED:
            interaction = self.interaction_service.update_consumed(
                interaction.interaction_id,
                InteractionResumePhase.RESULT_COMMITTED,
                checkpoint,
            )
        all_tool_results = [
            ToolResult.model_validate(item) for item in checkpoint.get("all_tool_results", [])
        ]
        if not any(item.tool_use_id == result.tool_use_id for item in all_tool_results):
            all_tool_results.append(result)
        return RuntimeTurnResult(
            session_id=interaction.session_id,
            run_id=interaction.run_id,
            status=RuntimeStatus.OK,
            tool_result_messages=[_tool_result_message(item) for item in all_tool_results],
            tool_results=all_tool_results,
            steps=interaction.step_index + 1,
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
            return _error_result(
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
            return _error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                metadata=run_metadata,
            )

        # --- 3. Persist result ---
        messages = [*history, current_input, assistant_message]
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
                    _build_run_metadata(
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
                _build_run_metadata(
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
                    _build_run_metadata(
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
            return _error_result(
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
                return _error_result(
                    session_id=session_id,
                    run_id=run_id,
                    error=normalize_runtime_error(exc),
                    assistant_message=latest_assistant,
                    steps=step_number,
                    metadata=run_metadata,
                )

            messages = [*history]
            if current_input is not None:
                messages.append(current_input)
            messages.append(latest_assistant)

            try:
                self.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )

                if not latest_assistant.has_tool_calls:
                    self.session_store.save_run_metadata(
                        session_id,
                        _build_run_metadata(
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
                        _build_run_metadata(
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
                return _error_result(
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
                error = _tool_error_info(bridge_result)
                try:
                    self.session_store.save_run_metadata(
                        session_id,
                        _build_run_metadata(
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
                    return _error_result(
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
                _build_run_metadata(
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
            return _error_result(
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


def _build_hitl_checkpoint(
    *,
    run_mode: str,
    agent_name: str,
    runtime_options: RuntimeOptions,
    assistant_message: Msg,
    response: LLMResponse,
    step_index: int,
    next_tool_index: int,
    batch_results: list[dict[str, Any]],
    all_tool_results: list[ToolResult],
    read_state: Any | None,
    call_fingerprint: str,
) -> dict[str, Any]:
    """构造并验证等待恢复所需的 JSON-safe checkpoint。"""
    import json

    try:
        read_state_json: Any | None = None
        if read_state is not None:
            read_state_json = (
                read_state.model_dump(mode="json")
                if hasattr(read_state, "model_dump")
                else read_state
            )
        checkpoint = RuntimeHITLCheckpoint(
            run_mode=run_mode,
            agent_name=agent_name,
            session_id=runtime_options.session_id,
            run_id=runtime_options.run_id,
            step_index=step_index,
            runtime_options=RuntimeOptionsSnapshot(options=runtime_options.model_dump(mode="json")),
            assistant_message={
                "role": "assistant",
                "content": [block.model_dump(mode="json") for block in response.content],
                "metadata": {
                    "provider": response.provider,
                    "id": response.id,
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                },
            },
            provider_response=ProviderResponseSnapshot(
                provider=response.provider,
                response_id=response.id,
                model=response.model,
                content=[block.model_dump(mode="json") for block in response.content],
                finish_reason=response.finish_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                reasoning=response.reasoning,
            ),
            tool_calls=[call.model_dump(mode="json") for call in assistant_message.tool_calls],
            next_tool_index=next_tool_index,
            batch_results=batch_results,
            all_tool_results=[result.model_dump(mode="json") for result in all_tool_results],
            read_state=read_state_json,
            call_fingerprint=call_fingerprint,
        ).model_dump(mode="json")
        json.dumps(checkpoint, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise HITLCheckpointInvalidError("HITL checkpoint 必须是 JSON-safe 数据") from exc
    return checkpoint


def _call_fingerprint(
    *,
    request: PermissionInteractionRequest | QuestionInteractionRequest,
    session_id: str,
    run_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    workspace_root: Path,
) -> str:
    """返回权限请求原有或问题请求派生的调用指纹。"""
    if isinstance(request, PermissionInteractionRequest):
        return request.call_fingerprint
    return make_call_fingerprint(
        session_id=session_id,
        run_id=run_id,
        tool_call_id=request.tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
        workspace_root=str(workspace_root),
    )


def _tool_result_message(result: ToolResult) -> Msg:
    """将结构化工具结果转换为 provider-neutral 消息。"""
    return Msg.tool_result(
        tool_use_id=result.tool_use_id,
        content=result.model_content,
        is_error=result.is_error,
        name=result.tool_name,
        metadata=result.to_block_metadata(),
    )


def normalize_runtime_error(error: Exception) -> RuntimeErrorInfo:
    """将 runtime 边界异常归一化为稳定错误信息。

    Args:
        error (Exception): Runtime 边界捕获到的异常。

    Returns:
        RuntimeErrorInfo: 可放入 `RuntimeTurnResult.error` 的结构化错误。
    """
    code, source = _classify_runtime_error(error)
    details: dict[str, Any] = {}
    if isinstance(error, IrisError):
        details.update(error.context)
    return RuntimeErrorInfo(
        code=code,
        message=str(error),
        source=source,
        details=details,
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


def _build_run_metadata(
    *,
    existing: dict[str, object],
    session_id: str,
    run_id: str,
    status: RuntimeStatus,
    provider: str,
    response: LLMResponse | None,
    message_count: int,
    metadata: Mapping[str, Any],
    steps: int = 1,
    tool_count: int = 0,
    error: RuntimeErrorInfo | None = None,
    waiting_human: bool = False,
    interaction_id: str | None = None,
) -> dict[str, object]:
    """构建 session 中保存的 run metadata。

    `latest_run` 便于快速读取最近一次结果，`runs` 保留 append-like 历史，避免 Stage 03
    引入新的持久化表结构或 session model。
    """
    latest_run: dict[str, object] = {
        "session_id": session_id,
        "run_id": run_id,
        "status": status.value,
        "provider": cast(object, provider),
        "model": cast(object, response.model if response is not None else ""),
        "finish_reason": cast(
            object,
            response.finish_reason if response is not None else "",
        ),
        "input_tokens": response.input_tokens if response is not None else 0,
        "output_tokens": response.output_tokens if response is not None else 0,
        "total_tokens": response.total_tokens if response is not None else 0,
        "message_count": message_count,
        "steps": steps,
        "tool_count": tool_count,
        "metadata": dict(metadata),
    }
    if error is not None:
        latest_run["error"] = error.model_dump(mode="json")
    if waiting_human:
        latest_run["waiting_human"] = True
    if interaction_id is not None:
        latest_run["interaction_id"] = interaction_id
    runs = existing.get("runs", [])
    run_list = list(runs) if isinstance(runs, list) else []
    run_list.append(latest_run)
    return {**existing, "latest_run": latest_run, "runs": run_list}


def _build_resume_run_metadata(
    *,
    existing: dict[str, object],
    result: RuntimeTurnResult,
    message_count: int,
) -> dict[str, object]:
    """基于原 run snapshot 构建 resume 后的 append-like metadata。"""
    previous = existing.get("latest_run")
    latest_run = dict(previous) if isinstance(previous, dict) else {}
    latest_run.update(
        session_id=result.session_id,
        run_id=result.run_id,
        status=result.status.value,
        message_count=message_count,
        steps=result.steps,
        tool_count=len(result.tool_results),
    )
    latest_run.pop("waiting_human", None)
    latest_run.pop("interaction_id", None)
    latest_run.pop("error", None)
    if result.error is not None:
        latest_run["error"] = result.error.model_dump(mode="json")
    if result.status is RuntimeStatus.WAITING_HUMAN:
        if result.pending_interaction is None:
            raise HITLCheckpointInvalidError("waiting_human 结果缺少 pending interaction")
        latest_run["waiting_human"] = True
        latest_run["interaction_id"] = result.pending_interaction.interaction_id
    runs = existing.get("runs", [])
    run_list = list(runs) if isinstance(runs, list) else []
    run_list.append(latest_run)
    return {**existing, "latest_run": latest_run, "runs": run_list}


def _error_result(
    *,
    session_id: str,
    run_id: str,
    error: RuntimeErrorInfo,
    assistant_message: Msg | None = None,
    steps: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> RuntimeTurnResult:
    """构造统一失败结果。"""
    return RuntimeTurnResult(
        session_id=session_id,
        run_id=run_id,
        status=RuntimeStatus.ERROR,
        assistant_message=assistant_message,
        steps=steps,
        error=error,
        metadata=dict(metadata or {}),
    )


def _should_stop_on_tool_error(
    options: RuntimeOptions,
    bridge_result: ToolBridgeResult,
) -> bool:
    """判断 loop 是否应在工具错误后停止。"""
    return options.loop.tool_error_policy == ToolErrorPolicy.STOP and any(
        result.is_error for result in bridge_result.results
    )


def _tool_error_info(bridge_result: ToolBridgeResult) -> RuntimeErrorInfo:
    """从第一个工具错误构造 runtime 错误信息。

    并行执行时不一定是调用顺序上的第一个失败
    """
    for result in bridge_result.results:
        if result.is_error and result.error is not None:
            return RuntimeErrorInfo(
                code=result.error.code,
                message=result.error.message,
                source="tool",
                details=result.error.details,
            )
    return RuntimeErrorInfo(
        code="TOOL_ERROR",
        message="工具执行失败",
        source="tool",
    )


def _classify_runtime_error(error: Exception) -> tuple[str, RuntimeErrorSource]:
    """从 Iris 异常实例读取 runtime 错误映射。"""
    if isinstance(error, IrisError):
        return error.runtime_code, error.runtime_source
    return "RUNTIME_ERROR", "runtime"


__all__ = ["AgentRuntime", "RuntimeProvider", "normalize_runtime_error"]
