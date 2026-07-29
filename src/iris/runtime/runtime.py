"""单轮 Agent runtime。

本模块从 session history、context 和当前用户输入构造一次 `LLMRequest`，
调用注入的 provider，并把 assistant 回复写回 session。
memory 只支持显式 opt-in 注入；工具执行只做一次 bridge，bounded loop 留给后续阶段组合。

Example:
    runtime = AgentRuntime(environment)
    result = await runtime.run_turn("你好")
"""

# region imports
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from ..exceptions import (
    HITLCheckpointInvalidError,
    HITLExecutionOutcomeUnknownError,
    HITLResponseRequiredError,
    IrisError,
    IrisRunConflictError,
)
from ..hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
)
from ..lifecycle import RunErrorInfo
from ..message import ContentBlock, LLMRequest, LLMResponse, Msg, ToolUseBlock
from ..session import SessionStore
from ..tools import (
    CancellationRequestedError,
    CancellationSignal,
    PreparedToolCall,
    ToolExecutionContext,
    ToolRegistryView,
    ToolResult,
)
from .checkpoint import build_hitl_checkpoint, validate_hitl_checkpoint
from .commit import (
    CommitPortToolEffectGuard,
    RuntimeCommitPort,
    RuntimeModelStepCommit,
    RuntimeSuspension,
    RuntimeSuspensionResult,
    RuntimeToolResultCommit,
    build_runtime_tool_call,
)
from .environment import RuntimeEnvironment
from .errors import error_result, normalize_runtime_error, tool_error_info
from .memory_context import (
    prepare_activation_memory_context_input,
    prepare_memory_context_input,
)
from .metadata import build_run_metadata, synchronize_resume_metadata
from .models import (
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeContinuationClaim,
    RuntimeCursor,
    RuntimeErrorInfo,
    RuntimeOptions,
    RuntimeStatus,
    RuntimeTurnResult,
    ToolErrorPolicy,
    ToolResultCommit,
)
from .resume import (
    append_resumed_result,
    commit_ready_interaction,
    load_resumable_interaction,
    resolve_interaction_result,
)
from .tool_result_committer import commit_tool_results

# endregion


class AgentRuntime:
    """编排一次本地 Agent 单轮调用。

    `AgentRuntime` 只负责调用顺序和边界错误归一化；context 构建、消息装配、
    provider 调用和 session 存储仍由各自组件承担。这样单轮 runtime 可先稳定下来，
    后续 memory、工具桥和 loop 可以在同一边界上继续组合。

    Example:
        runtime = AgentRuntime(environment)
        result = await runtime.run_turn("当前问题")
    """

    def __init__(self, environment: RuntimeEnvironment) -> None:
        """创建绑定完整构造期环境的 Agent runtime。

        Args:
            environment (RuntimeEnvironment): 已装配且与 runtime 同生命周期的依赖环境。
        """
        self.environment = environment

    async def execute(
        self,
        activation: RuntimeActivationInput,
        *,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
    ) -> RuntimeActivationResult:
        """从 durable cursor 推进唯一的 model/tool inner loop。"""
        cursor = activation.cursor
        self.environment.tool_bridge.restore_read_state(
            activation.session_id,
            cursor.read_state,
        )
        interaction_projection = activation.interaction_projection
        projection_validated = False

        while True:
            if cursor.position == "outcome_ready":
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.COMPLETED,
                    cursor=cursor,
                    assistant_message=cursor.assistant_message,
                )
            if _activation_cancelled(commits, cancellation):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.CANCELLED,
                    cursor=cursor,
                    assistant_message=cursor.assistant_message,
                )
            if _deadline_expired(commits):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                    cursor=cursor,
                    assistant_message=cursor.assistant_message,
                )

            if cursor.position == "before_model":
                model_outcome = await self._execute_model_step(
                    activation=activation,
                    cursor=cursor,
                    commits=commits,
                    cancellation=cancellation,
                )
                if isinstance(model_outcome, RuntimeActivationResult):
                    return model_outcome
                cursor = model_outcome
                continue

            plan = self.environment.tool_bridge.preflight_once(
                assistant_message=cast(Msg, cursor.assistant_message),
                session_id=activation.session_id,
                run_id=activation.run_id,
                agent_id=self.environment.agent_config.name,
                workspace_root=self.environment.workspace_root,
                permission_mode=self.environment.agent_config.permissions.writes,
                metadata={"activation_id": activation.activation_id},
                tools_enabled=activation.options.include_tools,
                cancellation=cancellation,
            )
            if not projection_validated:
                _validate_interaction_projection(
                    interaction_projection,
                    cursor,
                    plan.calls,
                )
                projection_validated = True
            prepared = plan.calls[cursor.next_tool_index]
            approved_projection: RuntimeApprovedToolCall | None = None
            projected_result: ToolResult | None = None
            if prepared.human_request is not None:
                if isinstance(interaction_projection, ToolResult):
                    projected_result = interaction_projection
                    interaction_projection = None
                elif isinstance(interaction_projection, RuntimeApprovedToolCall):
                    approved_projection = interaction_projection
                    interaction_projection = None
                else:
                    return self._suspend_existing_batch(
                        activation=activation,
                        cursor=cursor,
                        plan=plan.calls,
                        prepared=prepared,
                        commits=commits,
                    )

            if projected_result is not None:
                result = projected_result
                claim = None
                tool_call = build_runtime_tool_call(
                    activation=activation,
                    cursor=cursor,
                    prepared=prepared,
                    workspace_root=self.environment.workspace_root,
                )
            elif prepared.preflight_result is not None:
                result = prepared.preflight_result
                claim = None
                tool_call = build_runtime_tool_call(
                    activation=activation,
                    cursor=cursor,
                    prepared=prepared,
                    workspace_root=self.environment.workspace_root,
                )
            else:
                if _activation_cancelled(commits, cancellation):
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.CANCELLED,
                        cursor=cursor,
                        assistant_message=cursor.assistant_message,
                    )
                guard = CommitPortToolEffectGuard(
                    activation=activation,
                    cursor=cursor,
                    commits=commits,
                    workspace_root=self.environment.workspace_root,
                    interaction_id=(
                        approved_projection.interaction_id
                        if approved_projection is not None
                        else None
                    ),
                )
                timeout = _tool_timeout_seconds(activation, commits)
                try:
                    operation = self.environment.tool_bridge.execute_prepared(
                        prepared,
                        session_id=activation.session_id,
                        run_id=activation.run_id,
                        agent_id=self.environment.agent_config.name,
                        workspace_root=self.environment.workspace_root,
                        permission_mode=self.environment.agent_config.permissions.writes,
                        metadata={"activation_id": activation.activation_id},
                        cancellation=cancellation,
                        effect_guard=guard,
                        approved_tool_call_id=(
                            prepared.tool_use.id
                            if approved_projection is not None
                            else None
                        ),
                    )
                    result = (
                        await asyncio.wait_for(operation, timeout=timeout)
                        if timeout is not None
                        else await operation
                    )
                except CancellationRequestedError:
                    if guard.claim_for(prepared.tool_use.id) is not None:
                        return _unknown_tool_outcome(cursor, prepared, "工具 claim 后收到取消")
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.CANCELLED,
                        cursor=cursor,
                        assistant_message=cursor.assistant_message,
                    )
                except TimeoutError:
                    if guard.claim_for(prepared.tool_use.id) is not None:
                        return _unknown_tool_outcome(cursor, prepared, "工具 claim 后执行超时")
                    if _deadline_expired(commits):
                        return RuntimeActivationResult(
                            outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                            cursor=cursor,
                            assistant_message=cursor.assistant_message,
                        )
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.FAILED,
                        cursor=cursor,
                        assistant_message=cursor.assistant_message,
                        error=RunErrorInfo(
                            code="TOOL_TIMEOUT",
                            message="工具执行超时",
                            source="tool",
                        ),
                    )
                claim = guard.claim_for(prepared.tool_use.id)
                tool_call = guard.call_for(prepared.tool_use.id) or build_runtime_tool_call(
                    activation=activation,
                    cursor=cursor,
                    prepared=prepared,
                    workspace_root=self.environment.workspace_root,
                )

            result_message = _tool_result_message(result)
            batch_assistant = cursor.assistant_message
            next_index = cursor.next_tool_index + 1
            next_results = (*cursor.tool_results, result)
            if next_index == len(cursor.tool_calls):
                cursor_after = RuntimeCursor(
                    position="before_model",
                    step_index=cursor.step_index + 1,
                    read_state=_read_state_snapshot(
                        self.environment.tool_bridge.read_state(activation.session_id)
                    ),
                )
            else:
                cursor_after = cursor.model_copy(
                    update={
                        "next_tool_index": next_index,
                        "tool_results": next_results,
                        "read_state": _read_state_snapshot(
                            self.environment.tool_bridge.read_state(activation.session_id)
                        ),
                    }
                )
            committed_cursor = commits.commit_tool_result(
                RuntimeToolResultCommit(
                    tool_call=tool_call,
                    claim=claim,
                    result=result,
                    message_delta=(result_message,),
                    cursor_after=cursor_after,
                )
            )
            if committed_cursor != cursor_after:
                raise IrisRunConflictError("tool-result commit 返回了意外 cursor")
            cursor = committed_cursor
            if _activation_cancelled(commits, cancellation):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.CANCELLED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                )
            if (
                result.is_error
                and activation.options.tool_error_policy is ToolErrorPolicy.STOP
            ):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.FAILED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                    error=_tool_run_error(result),
                )

    async def _execute_model_step(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
    ) -> RuntimeCursor | RuntimeActivationResult:
        """执行并 required commit 一次 provider step。"""
        snapshot = commits.load_session()
        if snapshot.session_id != activation.session_id:
            raise IrisRunConflictError("commit port 返回了跨 session history")
        try:
            context_input = self.environment.context_input
            if cursor.step_index == 0:
                context_input = prepare_activation_memory_context_input(
                    context_input,
                    options=activation.options,
                    memory_service=self.environment.memory_service,
                    memory_context_builder=self.environment.memory_context_builder,
                )
            context_output = self.environment.context_builder.build(context_input)
            current_input = (
                Msg.user(cast(str, activation.input))
                if activation.kind == "start" and cursor.step_index == 0
                else None
            )
            turn_messages = self.environment.assembler.build_turn_messages(
                context_output=context_output,
                current_input=current_input,
            )
            request = self.environment.assembler.build_request(
                agent_config=self.environment.agent_config,
                context_output=context_output,
                history=list(snapshot.messages),
                current_input=current_input,
            )
            request = _apply_request_options(request, activation.options.request_options)
            request = _apply_tool_schemas(
                request,
                include_tools=activation.options.include_tools,
                tool_view=self.environment.tool_bridge.tool_view,
                provider=self.environment.agent_config.model.provider,
            )
        except Exception as exc:
            return _failed_activation(cursor, exc)

        if _activation_cancelled(commits, cancellation):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED,
                cursor=cursor,
            )
        reservation = commits.reserve_model_step(cursor)
        if not reservation.granted:
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.BUDGET_EXHAUSTED,
                cursor=cursor,
            )
        if reservation.cursor != cursor or reservation.step_index != cursor.step_index:
            raise IrisRunConflictError("model-step reservation 与当前 cursor 不匹配")
        remaining = reservation.remaining_deadline_seconds
        if remaining is not None and remaining <= 0:
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                cursor=cursor,
            )
        if _activation_cancelled(commits, cancellation):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED,
                cursor=cursor,
            )

        try:
            operation = self.environment.provider.complete(request)
            response = (
                await asyncio.wait_for(operation, timeout=remaining)
                if remaining is not None
                else await operation
            )
            assistant = response.to_msg()
        except TimeoutError as exc:
            if remaining is not None and _deadline_expired(commits):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                    cursor=cursor,
                )
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.FAILED,
                cursor=cursor,
                error=RunErrorInfo(
                    code="PROVIDER_TIMEOUT",
                    message=str(exc) or "Provider operation timeout",
                    source="provider",
                ),
            )
        except Exception as exc:
            return _failed_activation(cursor, exc)

        if _activation_cancelled(commits, cancellation):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED,
                cursor=cursor,
            )
        plan = self.environment.tool_bridge.preflight_once(
            assistant_message=assistant,
            session_id=activation.session_id,
            run_id=activation.run_id,
            agent_id=self.environment.agent_config.name,
            workspace_root=self.environment.workspace_root,
            permission_mode=self.environment.agent_config.permissions.writes,
            metadata={"activation_id": activation.activation_id},
            tools_enabled=activation.options.include_tools,
            cancellation=cancellation,
        )
        message_delta = (*turn_messages, assistant)
        read_state = _read_state_snapshot(
            self.environment.tool_bridge.read_state(activation.session_id)
        )
        if not assistant.tool_calls:
            cursor_after = RuntimeCursor(
                position="outcome_ready",
                step_index=cursor.step_index,
                assistant_message=assistant,
                read_state=read_state,
            )
            committed = commits.commit_model_step(
                RuntimeModelStepCommit(
                    cursor_before=cursor,
                    message_delta=message_delta,
                    assistant_message=assistant,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    cursor_after=cursor_after,
                )
            )
            if committed != cursor_after:
                raise IrisRunConflictError("model-step commit 返回了意外 cursor")
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.COMPLETED,
                cursor=committed,
                assistant_message=assistant,
            )

        cursor_after = RuntimeCursor(
            position="tool_batch",
            step_index=cursor.step_index,
            tool_calls=tuple(assistant.tool_calls),
            assistant_message=assistant,
            read_state=read_state,
        )
        prepared_facts = tuple(
            build_runtime_tool_call(
                activation=activation,
                cursor=cursor,
                prepared=prepared,
                workspace_root=self.environment.workspace_root,
                ordinal=index,
            )
            for index, prepared in enumerate(plan.calls, start=1)
        )
        gate = plan.first_human_gate
        if gate is not None and gate.human_request is not None:
            suspended = commits.suspend(
                RuntimeSuspension(
                    cursor_before=cursor,
                    message_delta=message_delta,
                    assistant_message=assistant,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    prepared_tool_calls=prepared_facts,
                    cursor=cursor_after,
                    interaction_request=gate.human_request,
                )
            )
            _validate_suspension_projection(
                activation=activation,
                cursor=cursor_after,
                interaction_request=gate.human_request,
                suspended=suspended,
            )
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.SUSPENDED,
                cursor=suspended.cursor,
                assistant_message=assistant,
                suspension=suspended.interaction,
            )
        committed = commits.commit_model_step(
            RuntimeModelStepCommit(
                cursor_before=cursor,
                message_delta=message_delta,
                assistant_message=assistant,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                prepared_tool_calls=prepared_facts,
                cursor_after=cursor_after,
            )
        )
        if committed != cursor_after:
            raise IrisRunConflictError("model-step commit 返回了意外 cursor")
        return committed

    def _suspend_existing_batch(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        plan: Sequence[PreparedToolCall],
        prepared: PreparedToolCall,
        commits: RuntimeCommitPort,
    ) -> RuntimeActivationResult:
        """在已提交 model step 的下一处 gate 原子暂停。"""
        if prepared.human_request is None or cursor.assistant_message is None:
            raise HITLCheckpointInvalidError("tool batch gate 缺少 interaction fact")
        prepared_facts = tuple(
            build_runtime_tool_call(
                activation=activation,
                cursor=cursor,
                prepared=item,
                workspace_root=self.environment.workspace_root,
                ordinal=index,
            )
            for index, item in enumerate(
                plan[cursor.next_tool_index :],
                start=cursor.next_tool_index + 1,
            )
        )
        suspended = commits.suspend(
            RuntimeSuspension(
                cursor_before=cursor,
                assistant_message=cursor.assistant_message,
                prepared_tool_calls=prepared_facts,
                cursor=cursor,
                interaction_request=prepared.human_request,
            )
        )
        _validate_suspension_projection(
            activation=activation,
            cursor=cursor,
            interaction_request=prepared.human_request,
            suspended=suspended,
        )
        return RuntimeActivationResult(
            outcome=RuntimeActivationOutcome.SUSPENDED,
            cursor=suspended.cursor,
            assistant_message=cursor.assistant_message,
            suspension=suspended.interaction,
        )

    def _create_waiting_interaction(
        self,
        *,
        run_mode: Literal["turn", "loop"],
        assistant_message: Msg,
        response: LLMResponse,
        runtime_options: RuntimeOptions,
        metadata: Mapping[str, Any],
        step_index: int,
        all_tool_results: list[ToolResult],
    ) -> HumanInteraction | None:
        """预检工具批次，并在第一处人工 gate 创建持久化 interaction。"""
        plan = self.environment.tool_bridge.preflight_once(
            assistant_message=assistant_message,
            session_id=runtime_options.session_id,
            run_id=runtime_options.run_id,
            agent_id=self.environment.agent_config.name,
            workspace_root=self.environment.workspace_root,
            permission_mode=self.environment.agent_config.permissions.writes,
            metadata=metadata,
            tools_enabled=runtime_options.include_tools,
        )
        gate = plan.first_human_gate
        if gate is None or gate.human_request is None:
            return None

        checkpoint = build_hitl_checkpoint(
            run_mode=run_mode,
            agent_name=self.environment.agent_config.name,
            runtime_options=runtime_options,
            assistant_message=assistant_message,
            response=response,
            step_index=step_index,
            next_tool_index=0,
            batch_results=[],
            all_tool_results=all_tool_results,
            read_state=self.environment.tool_bridge.read_state(runtime_options.session_id),
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
        return self.environment.interaction_service.create(
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
            session_store=self.environment.session_store,
            interaction_service=self.environment.interaction_service,
            session_id=session_id,
        )

    async def resume(
        self,
        interaction_id: str,
        response: PermissionInteractionResponse | QuestionInteractionResponse | None = None,
    ) -> RuntimeTurnResult:
        """恢复已等待人工响应的一次工具调用。"""
        try:
            interaction = self.environment.interaction_service.get(interaction_id)
            if interaction.status is InteractionStatus.PENDING:
                if response is None:
                    raise HITLResponseRequiredError("HITL interaction 尚未收到 response")
                interaction = self.environment.interaction_service.resolve(interaction_id, response)
            elif interaction.status is InteractionStatus.RESOLVED and response is not None:
                interaction = self.environment.interaction_service.resolve(interaction_id, response)
            checkpoint, options = validate_hitl_checkpoint(
                interaction,
                agent_name=self.environment.agent_config.name,
            )
            self.environment.tool_bridge.restore_read_state(
                interaction.session_id,
                (
                    checkpoint.get("read_state")
                    if isinstance(checkpoint.get("read_state"), dict)
                    else None
                ),
            )
            calls: list[ContentBlock] = [
                ToolUseBlock.model_validate(item) for item in checkpoint["tool_calls"]
            ]
            plan = self.environment.tool_bridge.preflight_once(
                assistant_message=Msg.assistant(calls),
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                agent_id=self.environment.agent_config.name,
                workspace_root=self.environment.workspace_root,
                permission_mode=self.environment.agent_config.permissions.writes,
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
                    session_store=self.environment.session_store,
                    interaction_service=self.environment.interaction_service,
                    agent_id=self.environment.agent_config.name,
                )
                interaction = self.environment.interaction_service.get(interaction_id)
                if checkpoint.get("continuation_complete") is True:
                    return synchronize_resume_metadata(
                        session_store=self.environment.session_store,
                        result=committed,
                    )
                return synchronize_resume_metadata(
                    session_store=self.environment.session_store,
                    result=await self._resume_batch(
                        committed=committed,
                        interaction=interaction,
                        checkpoint=checkpoint,
                        options=options,
                        plan=plan,
                        next_index=int(checkpoint["next_tool_index"]),
                    ),
                )

            interaction = self.environment.interaction_service.claim(interaction_id, checkpoint)
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
                tool_executor=self.environment.tool_bridge.tool_executor,
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
            interaction = self.environment.interaction_service.update_consumed(
                interaction_id,
                InteractionResumePhase.RESULT_READY,
                ready_checkpoint,
                expected_phase=interaction.resume_phase,
                expected_version=interaction.version,
            )
            committed = await commit_ready_interaction(
                interaction=interaction,
                session_store=self.environment.session_store,
                interaction_service=self.environment.interaction_service,
                agent_id=self.environment.agent_config.name,
            )
            interaction = self.environment.interaction_service.get(interaction_id)
            return synchronize_resume_metadata(
                session_store=self.environment.session_store,
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
            loaded_interaction = self.environment.interaction_service.store.load_interaction(
                interaction_id
            )
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
                session_store=self.environment.session_store,
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
            result = await self.environment.tool_bridge.tool_executor.execute_prepared(
                prepared, self._tool_context(options)
            )
            message = append_resumed_result(
                result=result,
                session_store=self.environment.session_store,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
                agent_id=self.environment.agent_config.name,
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
                result = await self.environment.tool_bridge.tool_executor.execute_prepared(
                    prepared,
                    self._tool_context(options),
                )
            append_resumed_result(
                result=result,
                session_store=self.environment.session_store,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                step_index=interaction.step_index,
                agent_id=self.environment.agent_config.name,
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
                if (state := self.environment.tool_bridge.read_state(interaction.session_id))
                is not None
                else None
            ),
            continuation_complete=continuation_complete,
        )
        interaction = self.environment.interaction_service.update_consumed(
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
        interaction = self.environment.interaction_service.update_consumed(
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
                if (state := self.environment.tool_bridge.read_state(interaction.session_id))
                is not None
                else None
            ),
            continuation_claim=None,
            continuation_complete=continuation_complete,
        )
        interaction = self.environment.interaction_service.update_consumed(
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
        read_state = self.environment.tool_bridge.read_state(interaction.session_id)
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
            workspace_root=self.environment.workspace_root,
            session_id=options.session_id,
            agent_id=self.environment.agent_config.name,
            permission_mode=self.environment.agent_config.permissions.writes,
            metadata={**options.metadata, "run_id": options.run_id},
            read_state=self.environment.tool_bridge.read_state(options.session_id),
        )

    async def _execute_and_commit_tool_results(
        self,
        *,
        assistant_message: Msg,
        session_id: str,
        run_id: str,
        step_index: int,
        metadata: Mapping[str, Any] | None,
        tools_enabled: bool,
    ) -> ToolResultCommit:
        """执行 assistant 工具调用并统一提交结果。"""
        results = await self.environment.tool_bridge.execute_once(
            assistant_message=assistant_message,
            session_id=session_id,
            agent_id=self.environment.agent_config.name,
            workspace_root=self.environment.workspace_root,
            permission_mode=self.environment.agent_config.permissions.writes,
            metadata=metadata,
            tools_enabled=tools_enabled,
        )
        return commit_tool_results(
            results=results,
            session_store=self.environment.session_store,
            session_id=session_id,
            run_id=run_id,
            step_index=step_index,
            agent_id=self.environment.agent_config.name,
            metadata=metadata,
            deduplicate_messages=False,
        )

    async def _continue_resumed_loop(
        self,
        committed: RuntimeTurnResult,
        options: RuntimeOptions,
    ) -> RuntimeTurnResult:
        """将已提交的工具结果回灌 provider，继续被暂停的 loop。"""
        try:
            history = _load_history(self.environment.session_store, committed.session_id)
            context_output = self.environment.context_builder.build(
                prepare_memory_context_input(
                    self.environment.context_input,
                    options=options,
                    memory_service=self.environment.memory_service,
                    memory_context_builder=self.environment.memory_context_builder,
                )
            )
            request = self.environment.assembler.build_request(
                agent_config=self.environment.agent_config,
                context_output=context_output,
                history=history,
                current_input=None,
            )
            request = _apply_tool_schemas(
                _apply_request_options(request, options.request_options),
                include_tools=options.include_tools,
                tool_view=self.environment.tool_bridge.tool_view,
                provider=self.environment.agent_config.model.provider,
            )
            response = await self.environment.provider.complete(request)
            assistant = response.to_msg()
            history.append(assistant)
            self.environment.session_store.save_messages(
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
            bridge = await self._execute_and_commit_tool_results(
                assistant_message=assistant,
                session_id=committed.session_id,
                run_id=committed.run_id,
                step_index=committed.steps,
                metadata=options.metadata,
                tools_enabled=options.include_tools,
            )
            if bridge.messages:
                history.extend(bridge.messages)
            continued = committed.model_copy(
                update={
                    "assistant_message": assistant,
                    "tool_results": [*committed.tool_results, *bridge.results],
                    "tool_result_messages": [*committed.tool_result_messages, *bridge.messages],
                    "steps": committed.steps + 1,
                }
            )
            if _should_stop_on_tool_error(options, bridge.results):
                return RuntimeTurnResult(
                    session_id=continued.session_id,
                    run_id=continued.run_id,
                    status=RuntimeStatus.ERROR,
                    assistant_message=assistant,
                    tool_results=continued.tool_results,
                    tool_result_messages=continued.tool_result_messages,
                    steps=continued.steps,
                    error=tool_error_info(bridge.results),
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
            history = _load_history(self.environment.session_store, session_id)
            context_input = prepare_memory_context_input(
                self.environment.context_input,
                options=runtime_options,
                memory_service=self.environment.memory_service,
                memory_context_builder=self.environment.memory_context_builder,
            )
            context_output = self.environment.context_builder.build(context_input)
            current_input = Msg.user(user_input)
            turn_messages = self.environment.assembler.build_turn_messages(
                context_output=context_output,
                current_input=current_input,
            )
            request = self.environment.assembler.build_request(
                agent_config=self.environment.agent_config,
                context_output=context_output,
                history=history,
                current_input=current_input,
            )
            request = _apply_request_options(request, runtime_options.request_options)
            request = _apply_tool_schemas(
                request,
                include_tools=runtime_options.include_tools,
                tool_view=self.environment.tool_bridge.tool_view,
                provider=self.environment.agent_config.model.provider,
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
            response = await self.environment.provider.complete(request)
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
            self.environment.session_store.save_messages(
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
                self.environment.session_store.save_run_metadata(
                    session_id,
                    build_run_metadata(
                        existing=self.environment.session_store.load_run_metadata(session_id),
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.WAITING_HUMAN,
                        provider=self.environment.agent_config.model.provider,
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
            bridge_result = await self._execute_and_commit_tool_results(
                assistant_message=assistant_message,
                session_id=session_id,
                run_id=run_id,
                step_index=0,
                metadata=run_metadata,
                tools_enabled=runtime_options.include_tools,
            )
            if bridge_result.messages:
                messages.extend(bridge_result.messages)
            self.environment.session_store.save_run_metadata(
                session_id,
                build_run_metadata(
                    existing=self.environment.session_store.load_run_metadata(session_id),
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.OK,
                    provider=self.environment.agent_config.model.provider,
                    response=response,
                    message_count=len(messages),
                    metadata=run_metadata,
                    tool_count=len(bridge_result.results),
                ),
            )
        except Exception as exc:
            error = normalize_runtime_error(exc)
            try:
                self.environment.session_store.save_run_metadata(
                    session_id,
                    build_run_metadata(
                        existing=self.environment.session_store.load_run_metadata(session_id),
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.ERROR,
                        provider=self.environment.agent_config.model.provider,
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
                history = _load_history(self.environment.session_store, session_id)
                context_input = prepare_memory_context_input(
                    self.environment.context_input,
                    options=runtime_options,
                    memory_service=self.environment.memory_service,
                    memory_context_builder=self.environment.memory_context_builder,
                )
                context_output = self.environment.context_builder.build(context_input)
                turn_messages = self.environment.assembler.build_turn_messages(
                    context_output=context_output,
                    current_input=current_input,
                )
                request = self.environment.assembler.build_request(
                    agent_config=self.environment.agent_config,
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
                    tool_view=self.environment.tool_bridge.tool_view,
                    provider=self.environment.agent_config.model.provider,
                )
                latest_response = await self.environment.provider.complete(request)
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
                self.environment.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )

                if not latest_assistant.has_tool_calls:
                    self.environment.session_store.save_run_metadata(
                        session_id,
                        build_run_metadata(
                            existing=self.environment.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.OK,
                            provider=self.environment.agent_config.model.provider,
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
                    self.environment.session_store.save_run_metadata(
                        session_id,
                        build_run_metadata(
                            existing=self.environment.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.WAITING_HUMAN,
                            provider=self.environment.agent_config.model.provider,
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

                bridge_result = await self._execute_and_commit_tool_results(
                    assistant_message=latest_assistant,
                    session_id=session_id,
                    run_id=run_id,
                    step_index=step_index,
                    metadata=run_metadata,
                    tools_enabled=runtime_options.include_tools,
                )
                messages.extend(bridge_result.messages)
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

            if _should_stop_on_tool_error(runtime_options, bridge_result.results):
                error = tool_error_info(bridge_result.results)
                try:
                    self.environment.session_store.save_run_metadata(
                        session_id,
                        build_run_metadata(
                            existing=self.environment.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.ERROR,
                            provider=self.environment.agent_config.model.provider,
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
            self.environment.session_store.save_run_metadata(
                session_id,
                build_run_metadata(
                    existing=self.environment.session_store.load_run_metadata(session_id),
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.MAX_STEPS,
                    provider=self.environment.agent_config.model.provider,
                    response=latest_response,
                    message_count=len(self.environment.session_store.load_messages(session_id)),
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


def _validate_interaction_projection(
    projection: ToolResult | RuntimeApprovedToolCall | None,
    cursor: RuntimeCursor,
    plan: Sequence[PreparedToolCall],
) -> None:
    """在任何批次 effect 前把 response projection 绑定到第一处未提交 gate。"""
    if projection is None:
        return
    gate = next(
        (
            prepared
            for prepared in plan[cursor.next_tool_index :]
            if prepared.human_request is not None
        ),
        None,
    )
    if gate is None or gate.human_request is None:
        raise IrisRunConflictError("interaction projection 没有对应的 pending gate")
    snapshot = gate.human_request.tool_call
    if isinstance(projection, RuntimeApprovedToolCall):
        if not isinstance(gate.human_request.prompt, PermissionPrompt):
            raise IrisRunConflictError("interaction projection 类型与 question gate 不匹配")
        if (
            projection.tool_call_id != snapshot.tool_call_id
            or projection.tool_name != snapshot.tool_name
            or projection.fingerprint != snapshot.fingerprint
        ):
            raise IrisRunConflictError("interaction projection 与 pending gate 不匹配")
        return
    if (
        projection.tool_use_id != snapshot.tool_call_id
        or projection.tool_name != snapshot.tool_name
    ):
        raise IrisRunConflictError("interaction projection 与 pending gate 不匹配")
    if isinstance(gate.human_request.prompt, QuestionPrompt):
        if projection.is_error:
            raise IrisRunConflictError("question interaction projection 必须是回答结果")
        return
    if (
        not projection.is_error
        or projection.error is None
        or projection.error.code != "USER_REJECTED"
    ):
        raise IrisRunConflictError("permission interaction projection 必须是批准或拒绝")


def _activation_cancelled(
    commits: RuntimeCommitPort,
    cancellation: CancellationSignal,
) -> bool:
    """合并 activation live signal 与 durable cancellation request。"""
    return cancellation.requested or commits.cancellation_requested()


def _deadline_expired(commits: RuntimeCommitPort) -> bool:
    """判断 lifecycle owner 提供的 deadline 是否已经耗尽。"""
    remaining = commits.remaining_deadline_seconds()
    return remaining is not None and remaining <= 0


def _tool_timeout_seconds(
    activation: RuntimeActivationInput,
    commits: RuntimeCommitPort,
) -> float | None:
    """用 run deadline 收紧单次工具 timeout。"""
    remaining = commits.remaining_deadline_seconds()
    configured = activation.options.tool_timeout_seconds
    if remaining is None:
        return configured
    if configured is None:
        return remaining
    return min(remaining, configured)


def _read_state_snapshot(state: Any | None) -> dict[str, Any] | None:
    """仅把 JSON-safe read state 放入 cursor。"""
    if state is None:
        return None
    if isinstance(state, Mapping):
        return dict(state)
    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise HITLCheckpointInvalidError("工具 read state 不是 JSON-safe object")


def _tool_result_message(result: ToolResult) -> Msg:
    """把结构化工具结果转换为 provider history 消息。"""
    return Msg.tool_result(
        tool_use_id=result.tool_use_id,
        content=result.model_content,
        is_error=result.is_error,
        name=result.tool_name,
        metadata=result.to_block_metadata(),
    )


def _normalize_run_error(error: Exception) -> RunErrorInfo:
    """将 runtime 异常归一化为 lifecycle error fact。"""
    if isinstance(error, IrisError):
        return RunErrorInfo(
            code=error.runtime_code,
            message=str(error),
            source=cast(Any, error.runtime_source),
            details=dict(error.context),
        )
    return RunErrorInfo(
        code="RUNTIME_ERROR",
        message=str(error),
        source="runtime",
    )


def _failed_activation(
    cursor: RuntimeCursor,
    error: Exception,
) -> RuntimeActivationResult:
    """构造未产生 required durable fact 的 engine failure。"""
    return RuntimeActivationResult(
        outcome=RuntimeActivationOutcome.FAILED,
        cursor=cursor,
        assistant_message=cursor.assistant_message,
        error=_normalize_run_error(error),
    )


def _tool_run_error(result: ToolResult) -> RunErrorInfo:
    """从已提交的首个工具错误构造 engine failure。"""
    if result.error is None:
        return RunErrorInfo(
            code="TOOL_ERROR",
            message="工具执行失败",
            source="tool",
        )
    return RunErrorInfo(
        code=result.error.code,
        message=result.error.message,
        source="tool",
        details=result.error.details,
    )


def _unknown_tool_outcome(
    cursor: RuntimeCursor,
    prepared: PreparedToolCall,
    message: str,
) -> RuntimeActivationResult:
    """构造 claim 已存在但缺少 durable result 的 unknown fact。"""
    return RuntimeActivationResult(
        outcome=RuntimeActivationOutcome.OUTCOME_UNKNOWN,
        cursor=cursor,
        assistant_message=cursor.assistant_message,
        error=RunErrorInfo(
            code="TOOL_OUTCOME_UNKNOWN",
            message=message,
            source="tool",
            details={"tool_call_id": prepared.tool_use.id},
        ),
    )


def _validate_suspension_projection(
    *,
    activation: RuntimeActivationInput,
    cursor: RuntimeCursor,
    interaction_request: HumanInteractionRequest,
    suspended: RuntimeSuspensionResult,
) -> None:
    """拒绝 commit port 返回的跨 identity waiting projection。"""
    interaction = suspended.interaction
    if (
        suspended.cursor != cursor
        or interaction.session_id != activation.session_id
        or interaction.run_id != activation.run_id
        or interaction.step_index != cursor.step_index
        or interaction.request != interaction_request
    ):
        raise IrisRunConflictError("suspension commit 返回了意外 projection")


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
    results: Sequence[ToolResult],
) -> bool:
    """判断 loop 是否应在工具错误后停止。"""
    return options.loop.tool_error_policy == ToolErrorPolicy.STOP and any(
        result.is_error for result in results
    )


__all__ = ["AgentRuntime", "normalize_runtime_error"]
