"""从 durable cursor 推进一次 activation 的 low-level Agent engine。"""

# region imports
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..exceptions import (
    HITLCheckpointInvalidError,
    IrisCancellationRequestedError,
    IrisError,
    IrisRunConflictError,
)
from ..hitl import (
    HumanInteractionRequest,
    PermissionPrompt,
    QuestionPrompt,
)
from ..lifecycle import CheckpointResumability, RunErrorInfo, ToolErrorPolicy
from ..message import LLMRequest, Msg
from ..tools import (
    CancellationSignal,
    PreparedToolCall,
    ToolRegistryView,
    ToolResult,
)
from .commit import (
    CommitPortToolEffectGuard,
    RuntimeCommitPort,
    RuntimeModelStepCommit,
    RuntimeSuspension,
    RuntimeSuspensionResult,
    RuntimeToolCall,
    RuntimeToolResultCommit,
    ToolCallClaim,
    build_runtime_tool_call,
)
from .environment import RuntimeEnvironment
from .memory_context import prepare_activation_memory_context_input
from .models import (
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeCursor,
)
from .steering import RuntimeSteeringPort
from .tool_bridge import ToolBridge

# endregion

_MAX_PARALLEL_TOOL_CALLS = 8
_logger = logging.getLogger(__name__)


class AgentRuntime:
    """执行一次 activation inner loop；不拥有 logical-run lifecycle。"""

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
        steering: RuntimeSteeringPort | None = None,
    ) -> RuntimeActivationResult:
        """按 durable cursor 分阶段推进唯一的 model/tool inner loop。

        阶段边界由 cursor 位置驱动，只有对应事实提交成功后才进入下一阶段。
        """
        # --- 1. 恢复 activation 现场 ---
        # 从 checkpoint 还原工具共享状态，并保留本次 resume 的 HITL 投影。
        cursor = activation.cursor
        self.environment.tool_bridge.restore_read_state(
            activation.session_id,
            cursor.read_state,
        )
        interaction_projection = activation.interaction_projection
        projection_validated = False

        while True:
            # --- 2. 收口 activation 状态 ---
            # 先兑现已提交的最终结果；未完成时再检查取消与截止时间。
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

            # --- 3. 执行模型阶段 ---
            # before_model 只推进一次模型调用，成功后转入工具批次或结果终态。
            if cursor.position == "before_model":
                model_outcome = await self._execute_model_step(
                    activation=activation,
                    cursor=cursor,
                    commits=commits,
                    cancellation=cancellation,
                    steering=steering,
                )
                if isinstance(model_outcome, RuntimeActivationResult):
                    return model_outcome
                cursor = model_outcome
                continue

            # --- 4. 预检工具批次 ---
            # 为当前 assistant 消息重建执行计划，并校验恢复投影与 cursor 是否一致。
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

            # --- 5. 执行安全并发窗口 ---
            # RETURN_TO_MODEL 仅并发连续安全调用，结果仍按模型原始顺序提交。
            if activation.options.tool_error_policy is ToolErrorPolicy.RETURN_TO_MODEL:
                window = _parallel_tool_window(
                    start=cursor.next_tool_index,
                    calls=plan.calls,
                    tool_bridge=self.environment.tool_bridge,
                )
                if window:
                    window_outcome = await self._execute_parallel_tool_window(
                        activation=activation,
                        cursor=cursor,
                        window=window,
                        commits=commits,
                        cancellation=cancellation,
                        steering=steering,
                    )
                    if isinstance(window_outcome, RuntimeActivationResult):
                        return window_outcome
                    cursor = window_outcome
                    continue

            # --- 6. 处理当前工具交互 ---
            # 串行路径先消费 HITL 投影；缺少人工决定时在当前批次挂起。
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

            # --- 7. 取得当前工具结果 ---
            # 优先复用投影或预检结果，否则在 effect guard 保护下执行真实工具。
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
                            prepared.tool_use.id if approved_projection is not None else None
                        ),
                    )
                    result = (
                        await asyncio.wait_for(operation, timeout=timeout)
                        if timeout is not None
                        else await operation
                    )
                except IrisCancellationRequestedError:
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

            # --- 8. 提交工具结果并收口 ---
            # durable commit 成功后才推进 cursor，再处理取消或 STOP 失败策略。
            batch_assistant = cursor.assistant_message
            cursor = await self._commit_tool_result(
                activation=activation,
                cursor=cursor,
                commits=commits,
                tool_call=tool_call,
                claim=claim,
                result=result,
                cancellation=cancellation,
                steering=steering,
            )
            if _activation_cancelled(commits, cancellation):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.CANCELLED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                )
            if result.is_error and activation.options.tool_error_policy is ToolErrorPolicy.STOP:
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.FAILED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                    error=_tool_run_error(result),
                )

    async def _execute_parallel_tool_window(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        window: Sequence[tuple[int, PreparedToolCall]],
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None,
    ) -> RuntimeCursor | RuntimeActivationResult:
        """执行一个有界安全窗口，并按模型 ordinal 提交结果。"""
        prepared_calls = [prepared for _, prepared in window]
        self.environment.tool_bridge._initialize_parallel_read_state(
            activation.session_id,
            prepared_calls,
        )
        guards = [
            CommitPortToolEffectGuard(
                activation=activation,
                cursor=cursor,
                commits=commits,
                workspace_root=self.environment.workspace_root,
                tool_index=tool_index,
            )
            for tool_index, _ in window
        ]
        tasks: list[asyncio.Task[ToolResult]] = []
        runtime_cancelled_tasks: set[asyncio.Task[ToolResult]] = set()
        timeout = _tool_timeout_seconds(activation, commits)
        for (_, prepared), guard in zip(window, guards, strict=True):
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
            )
            task_operation = (
                asyncio.wait_for(operation, timeout=timeout) if timeout is not None else operation
            )
            tasks.append(asyncio.create_task(task_operation))

        pending = set(tasks)
        try:
            while pending:
                done, remaining = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if any(task.cancelled() or task.exception() is not None for task in done):
                    runtime_cancelled_tasks.update(remaining)
                    for task in remaining:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
                pending = remaining
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        result_slots: list[ToolResult | BaseException | None] = []
        infrastructure_errors: list[tuple[int, BaseException]] = []
        for offset, task in enumerate(tasks):
            if task.cancelled():
                try:
                    task.result()
                except asyncio.CancelledError as child_cancellation:
                    result_slots.append(child_cancellation)
                    if task not in runtime_cancelled_tasks:
                        infrastructure_errors.append((offset, child_cancellation))
                continue
            exception = task.exception()
            if exception is None:
                result_slots.append(task.result())
                continue
            result_slots.append(exception)
            if not isinstance(exception, (IrisCancellationRequestedError, TimeoutError)):
                infrastructure_errors.append((offset, exception))

        if infrastructure_errors:
            _, infrastructure_error = min(
                infrastructure_errors,
                key=lambda item: item[0],
            )
            raise infrastructure_error

        settlement_exception = next(
            (
                slot
                for task, slot in zip(tasks, result_slots, strict=True)
                if isinstance(slot, BaseException) and task not in runtime_cancelled_tasks
            ),
            None,
        )
        committed_cursor = cursor
        interrupted = False
        for offset, slot in enumerate(result_slots):
            if isinstance(slot, BaseException):
                interrupted = True
                break
            if slot is None:
                raise IrisRunConflictError("并发工具窗口缺少已观察结果")
            _, prepared = window[offset]
            guard = guards[offset]
            tool_call = guard.call_for(prepared.tool_use.id) or build_runtime_tool_call(
                activation=activation,
                cursor=cursor,
                prepared=prepared,
                workspace_root=self.environment.workspace_root,
                ordinal=window[offset][0] + 1,
            )
            committed_cursor = await self._commit_tool_result(
                activation=activation,
                cursor=committed_cursor,
                commits=commits,
                tool_call=tool_call,
                claim=guard.claim_for(prepared.tool_use.id),
                result=slot,
                cancellation=cancellation,
                steering=steering,
            )

        if interrupted:
            committed_count = committed_cursor.next_tool_index - cursor.next_tool_index
            for offset in range(committed_count, len(window)):
                prepared = window[offset][1]
                if guards[offset].claim_for(prepared.tool_use.id) is not None:
                    return _unknown_tool_outcome(
                        committed_cursor,
                        prepared,
                        "并发工具窗口存在未提交的 claimed 调用",
                    )
            assert settlement_exception is not None
            if _activation_cancelled(commits, cancellation) or isinstance(
                settlement_exception,
                IrisCancellationRequestedError,
            ):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.CANCELLED,
                    cursor=committed_cursor,
                    assistant_message=cursor.assistant_message,
                )
            if _deadline_expired(commits):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                    cursor=committed_cursor,
                    assistant_message=cursor.assistant_message,
                )
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.FAILED,
                cursor=committed_cursor,
                assistant_message=cursor.assistant_message,
                error=RunErrorInfo(
                    code="TOOL_TIMEOUT",
                    message="工具执行超时",
                    source="tool",
                ),
            )

        if _activation_cancelled(commits, cancellation):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED,
                cursor=committed_cursor,
                assistant_message=cursor.assistant_message,
            )
        if _deadline_expired(commits):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                cursor=committed_cursor,
                assistant_message=cursor.assistant_message,
            )
        return committed_cursor

    async def _commit_tool_result(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        tool_call: RuntimeToolCall,
        claim: ToolCallClaim | None,
        result: ToolResult,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None,
    ) -> RuntimeCursor:
        """封装既有的单步 result commit 与 cursor 推进。"""
        next_index = cursor.next_tool_index + 1
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
                    "tool_results": (*cursor.tool_results, result),
                    "read_state": _read_state_snapshot(
                        self.environment.tool_bridge.read_state(activation.session_id)
                    ),
                }
            )
        steering_claim = None
        if (
            steering is not None
            and next_index == len(cursor.tool_calls)
            and not (
                result.is_error
                and activation.options.tool_error_policy is ToolErrorPolicy.STOP
            )
            and not _activation_cancelled(commits, cancellation)
            and not _deadline_expired(commits)
        ):
            claimed_input = await steering.claim(
                activation.run_id,
                activation.activation_id,
            )
            if claimed_input is not None:
                steering_claim = (steering, claimed_input)
        message_delta = (_tool_result_message(result),)
        if steering_claim is not None:
            _, claimed_input = steering_claim
            message_delta = (*message_delta, claimed_input.message)
        try:
            committed_cursor = commits.commit_tool_result(
                RuntimeToolResultCommit(
                    tool_call=tool_call,
                    claim=claim,
                    result=result,
                    message_delta=message_delta,
                    cursor_after=cursor_after,
                )
            )
            if committed_cursor != cursor_after:
                raise IrisRunConflictError("tool-result commit 返回了意外 cursor")
        except Exception:
            if steering_claim is not None:
                claimed_steering, claimed_input = steering_claim
                _settle_steering_input(
                    claimed_steering,
                    activation=activation,
                    submission_id=claimed_input.submission_id,
                    reason="commit_failed",
                )
            raise
        if steering_claim is not None:
            claimed_steering, claimed_input = steering_claim
            _settle_steering_input(
                claimed_steering,
                activation=activation,
                submission_id=claimed_input.submission_id,
            )
        return committed_cursor

    async def _execute_model_step(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None,
    ) -> RuntimeCursor | RuntimeActivationResult:
        """执行并 required commit 一次 provider step。"""
        snapshot = commits.load_session()
        if snapshot.session_id != activation.session_id:
            raise IrisRunConflictError("commit port 返回了跨 session history")
        try:
            context_input = self.environment.context_input
            if cursor.step_index == 0:
                context_input = await prepare_activation_memory_context_input(
                    context_input,
                    options=activation.options,
                    memory_service=self.environment.memory_service,
                    memory_context_builder=self.environment.memory_context_builder,
                )
            context_output = self.environment.context_builder.build(context_input)
            current_input = (
                Msg.user(activation.input)
                if activation.input is not None and cursor.step_index == 0
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
            steering_claim = None
            if steering is not None:
                if _activation_cancelled(commits, cancellation):
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.CANCELLED,
                        cursor=cursor,
                    )
                if _deadline_expired(commits):
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                        cursor=cursor,
                    )
                claimed_input = await steering.claim(
                    activation.run_id,
                    activation.activation_id,
                )
                if claimed_input is not None:
                    steering_claim = (steering, claimed_input)
            if steering_claim is not None:
                claimed_steering, claimed_input = steering_claim
                cursor_after = RuntimeCursor(
                    position="before_model",
                    step_index=cursor.step_index + 1,
                    read_state=read_state,
                )
                try:
                    committed = commits.commit_model_step(
                        RuntimeModelStepCommit(
                            cursor_before=cursor,
                            message_delta=(*message_delta, claimed_input.message),
                            assistant_message=assistant,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            total_tokens=response.total_tokens,
                            cursor_after=cursor_after,
                        )
                    )
                    if committed != cursor_after:
                        raise IrisRunConflictError("model-step commit 返回了意外 cursor")
                except Exception:
                    _settle_steering_input(
                        claimed_steering,
                        activation=activation,
                        submission_id=claimed_input.submission_id,
                        reason="commit_failed",
                    )
                    raise
                _settle_steering_input(
                    claimed_steering,
                    activation=activation,
                    submission_id=claimed_input.submission_id,
                )
                return committed

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
                    resumability=CheckpointResumability.OUTCOME_READY,
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


def _parallel_tool_window(
    *,
    start: int,
    calls: Sequence[PreparedToolCall],
    tool_bridge: ToolBridge,
) -> tuple[tuple[int, PreparedToolCall], ...]:
    """返回从 start 开始、至多八条的连续并发候选。"""
    window: list[tuple[int, PreparedToolCall]] = []
    stop = min(len(calls), start + _MAX_PARALLEL_TOOL_CALLS)
    for index in range(start, stop):
        prepared = calls[index]
        if not tool_bridge._is_parallel_candidate(prepared):
            break
        window.append((index, prepared))
    return tuple(window) if len(window) >= 2 else ()


def _settle_steering_input(
    steering: RuntimeSteeringPort,
    *,
    activation: RuntimeActivationInput,
    submission_id: str,
    reason: str | None = None,
) -> None:
    """隔离 steering settlement callback，不改变 durable commit 结果。"""
    try:
        if reason is None:
            steering.acknowledge(submission_id)
        else:
            steering.fail(submission_id, reason)
    except Exception:
        _logger.exception(
            "Steering settlement callback 失败",
            extra={
                "run_id": activation.run_id,
                "activation_id": activation.activation_id,
                "submission_id": submission_id,
            },
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


__all__ = ["AgentRuntime"]
