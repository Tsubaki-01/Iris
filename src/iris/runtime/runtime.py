"""从 durable cursor 推进一次 activation 的 low-level Agent engine。"""

# region imports
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, cast

from ..context import ContextBuildOutput
from ..exceptions import (
    HITLCheckpointInvalidError,
    IrisAPIConnectionError,
    IrisCancellationRequestedError,
    IrisContextCompactionError,
    IrisError,
    IrisMCPOutcomeUnknownError,
    IrisProviderStreamError,
    IrisProviderStreamInterruptedError,
    IrisRateLimitExceededError,
    IrisRunConflictError,
)
from ..hitl import (
    HumanInteractionRequest,
    PermissionPrompt,
    QuestionPrompt,
)
from ..lifecycle import (
    CheckpointResumability,
    RunErrorInfo,
    RuntimeExecutionOptions,
    SessionCompaction,
    SessionContextWindow,
    SessionSnapshot,
    TokenUsage,
    ToolErrorPolicy,
)
from ..message import (
    LLMRequest,
    LLMResponse,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    Msg,
    ToolUseBlock,
)
from ..tools import (
    CancellationSignal,
    PreparedToolCall,
    ReadFileState,
    ToolBatchPlan,
    ToolRegistryView,
    ToolResult,
)
from ..tools.subagent import ChildWaiting, SubagentParentCall, SubagentTool
from ._compaction_summary import (
    consume_summary_response,
    next_summary_batch,
    serialize_history,
)
from ._prompts import render_prompt
from .commit import (
    CommitPortToolEffectGuard,
    RuntimeCommitPort,
    RuntimeCompactionCommit,
    RuntimeModelStepCommit,
    RuntimeRunInputCommit,
    RuntimeSuspension,
    RuntimeSuspensionResult,
    RuntimeToolCall,
    RuntimeToolResultCommit,
    ToolCallClaim,
    build_runtime_tool_call,
)
from .compaction import project_history, protected_message_indices, select_compaction_end
from .environment import RuntimeEnvironment, streaming_provider_for
from .memory_context import load_context_windows, select_context_window
from .models import (
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeActivationResult,
    RuntimeApprovedToolCall,
    RuntimeCursor,
)
from .steering import RuntimeSteeringPort
from .streaming import (
    RuntimeEventSink,
    _LiveToolEffectGuard,
    _runtime_stream_event,
)
from .tool_bridge import ToolBridge

# endregion

_MAX_PARALLEL_TOOL_CALLS = 8
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ModelStepAdvance:
    """一次已提交模型步骤及其同进程工具计划。"""

    cursor: RuntimeCursor
    plan: ToolBatchPlan | None = None


@dataclass(frozen=True, slots=True)
class _ToolCompletion:
    """已确认工具结果及在收回结果前是否耗尽单次期限。"""

    result: ToolResult
    timed_out: bool


async def _execute_tool_with_timeout(
    operation: Awaitable[ToolResult], timeout: float | None
) -> _ToolCompletion:
    """有限 IO 可能延后响应取消，但不能把已过期限的调用误判为正常完成。"""
    budget = asyncio.timeout(timeout)
    async with budget:
        result = await operation
    return _ToolCompletion(result, budget.expired())


def _task_cancellation_pending() -> bool:
    """已收回的工具结果不清除当前执行任务尚未传播的取消意图。"""
    return cast(asyncio.Task[object], asyncio.current_task()).cancelling() > 0


class _RuntimeSinkEmissionError(Exception):
    """标记 model event sink error，使其绕过 provider error 归一化。"""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


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
        stream_sink: RuntimeEventSink | None = None,
    ) -> RuntimeActivationResult:
        """按 durable cursor 分阶段推进唯一的 model/tool inner loop。

        阶段边界由 cursor 位置驱动，只有对应事实提交成功后才进入下一阶段。

        Args:
            activation (RuntimeActivationInput): 本次推进的可信 activation 输入。
            commits (RuntimeCommitPort): Required durable fact 提交端口。
            cancellation (CancellationSignal): Activation-scope 取消信号。
            steering (RuntimeSteeringPort | None): 可选瞬时输入端口。
            stream_sink (RuntimeEventSink | None): 可选同步 live event sink。

        Returns:
            RuntimeActivationResult: 当前 activation 的 engine outcome。
        """
        # --- 1. 恢复 activation 现场 ---
        # 从 checkpoint 还原工具共享状态，并保留本次 resume 的 HITL 投影。
        cursor = activation.cursor
        self.environment.tool_bridge.restore_read_state(
            activation.session_id,
            cursor.read_state,
        )
        interaction_projection = activation.interaction_projection
        plan: ToolBatchPlan | None = None

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

            # --- 3. 归档本轮输入 ---
            # 读取与渲染只发生在输入阶段；提交后的恢复直接沿历史继续。
            if cursor.position == "before_input":
                input_outcome = await self._prepare_run_input(
                    activation=activation,
                    cursor=cursor,
                    commits=commits,
                    cancellation=cancellation,
                )
                if isinstance(input_outcome, RuntimeActivationResult):
                    return input_outcome
                cursor = input_outcome
                continue

            # --- 4. 执行模型阶段 ---
            # before_model 只推进一次模型调用，成功后转入工具批次或结果终态。
            if cursor.position == "before_model":
                model_outcome = await self._execute_model_step(
                    activation=activation,
                    cursor=cursor,
                    commits=commits,
                    cancellation=cancellation,
                    steering=steering,
                    stream_sink=stream_sink,
                )
                if isinstance(model_outcome, RuntimeActivationResult):
                    return model_outcome
                cursor = model_outcome.cursor
                plan = model_outcome.plan
                continue

            # --- 4. 预检工具批次 ---
            # 为当前 assistant 消息重建执行计划，并校验恢复投影与 cursor 是否一致。
            if plan is None:
                _emit_tool_preparing(
                    stream_sink,
                    activation=activation,
                    cursor=cursor,
                    tool_calls=cursor.tool_calls[cursor.next_tool_index :],
                    start_ordinal=cursor.next_tool_index + 1,
                )
                plan = self._prepare_tool_plan(
                    assistant_message=cast(Msg, cursor.assistant_message),
                    commits=commits,
                    interaction_projection=interaction_projection,
                    session_id=activation.session_id,
                    run_id=activation.run_id,
                    agent_id=self.environment.agent_config.name,
                    workspace_root=self.environment.workspace_root,
                    permission_mode=self.environment.agent_config.permissions.writes,
                    metadata={"activation_id": activation.activation_id},
                    tools_enabled=activation.options.include_tools,
                    cancellation=cancellation,
                )
            if interaction_projection is not None:
                prepared_subject = plan.calls[cursor.next_tool_index]
                _validate_interaction_projection(
                    interaction_projection,
                    build_runtime_tool_call(
                        activation=activation,
                        cursor=cursor,
                        prepared=prepared_subject,
                        workspace_root=self.environment.workspace_root,
                    ),
                    prepared_subject.human_request,
                )

            # --- 5. 执行安全并发窗口 ---
            # RETURN_TO_MODEL 仅并发连续安全调用，结果仍按模型原始顺序提交。
            if (
                interaction_projection is None
                and activation.options.tool_error_policy is ToolErrorPolicy.RETURN_TO_MODEL
            ):
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
                        stream_sink=stream_sink,
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
            # projection 已绑定当前 durable subject；刷新后的 ALLOW/DENY 不再产生 gate。
            if isinstance(interaction_projection, ToolResult):
                projected_result = interaction_projection
                interaction_projection = None
            elif isinstance(interaction_projection, RuntimeApprovedToolCall):
                approved_projection = interaction_projection
                interaction_projection = None
            elif prepared.human_request is not None:
                return self._suspend_existing_batch(
                    activation=activation,
                    cursor=cursor,
                    plan=plan.calls,
                    prepared=prepared,
                    commits=commits,
                )

            # --- 7. 取得当前工具结果 ---
            # 优先复用投影或预检结果，否则在 effect guard 保护下执行真实工具。
            subagent_call: SubagentParentCall | None = None
            tool_timed_out = False
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
            elif isinstance(prepared.tool, SubagentTool):
                call = SubagentParentCall(activation.run_id, prepared.tool_use.id)
                linked = commits.load_subagent_link(tool_call_id=prepared.tool_use.id)
                if stream_sink is not None and linked is None:
                    stream_sink.emit(
                        _runtime_stream_event(
                            "tool.started",
                            run_id=activation.run_id,
                            session_id=activation.session_id,
                            activation_id=activation.activation_id,
                            step_index=cursor.step_index,
                            tool_call_id=prepared.tool_use.id,
                            tool_name=prepared.tool_use.name,
                            tool_ordinal=cursor.next_tool_index + 1,
                        )
                    )
                outcome = await self.environment.tool_bridge.execute_subagent_prepared(
                    prepared,
                    session_id=activation.session_id,
                    run_id=activation.run_id,
                    agent_id=self.environment.agent_config.name,
                    workspace_root=self.environment.workspace_root,
                    permission_mode=self.environment.agent_config.permissions.writes,
                    metadata={"activation_id": activation.activation_id},
                    cancellation=cancellation,
                    approved_tool_call_id=prepared.tool_use.id
                    if approved_projection is not None
                    else None,
                    linked_continuation=linked is not None,
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
                if isinstance(outcome, ChildWaiting):
                    suspended = commits.rebind_subagent_proxy(
                        call=call, waiting=outcome, cursor=cursor
                    )
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.SUSPENDED,
                        cursor=suspended.cursor,
                        assistant_message=cursor.assistant_message,
                        suspension=suspended.interaction,
                    )
                result = outcome
                claim = None
                tool_call = build_runtime_tool_call(
                    activation=activation,
                    cursor=cursor,
                    prepared=prepared,
                    workspace_root=self.environment.workspace_root,
                )
                if commits.load_subagent_link(tool_call_id=prepared.tool_use.id) is not None:
                    subagent_call = call
            else:
                if _activation_cancelled(commits, cancellation):
                    return RuntimeActivationResult(
                        outcome=RuntimeActivationOutcome.CANCELLED,
                        cursor=cursor,
                        assistant_message=cursor.assistant_message,
                    )
                durable_guard = CommitPortToolEffectGuard(
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
                guard = (
                    _LiveToolEffectGuard(
                        guard=durable_guard,
                        sink=stream_sink,
                        activation=activation,
                        step_index=cursor.step_index,
                        tool_ordinal=cursor.next_tool_index + 1,
                    )
                    if stream_sink is not None
                    else durable_guard
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
                    completion = await _execute_tool_with_timeout(operation, timeout)
                    result, tool_timed_out = completion.result, completion.timed_out
                except IrisMCPOutcomeUnknownError as error:
                    return _unknown_tool_outcome(cursor, prepared, error.message)
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
                steering=None if tool_timed_out else steering,
                stream_sink=stream_sink,
                subagent_call=subagent_call,
            )
            if _task_cancellation_pending():
                raise asyncio.CancelledError
            if _activation_cancelled(commits, cancellation):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.CANCELLED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                )
            if _deadline_expired(commits):
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                )
            if tool_timed_out:
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.FAILED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                    error=RunErrorInfo(code="TOOL_TIMEOUT", message="工具执行超时", source="tool"),
                )
            if result.is_error and activation.options.tool_error_policy is ToolErrorPolicy.STOP:
                return RuntimeActivationResult(
                    outcome=RuntimeActivationOutcome.FAILED,
                    cursor=cursor,
                    assistant_message=batch_assistant,
                    error=_tool_run_error(result),
                )

    def _prepare_tool_plan(
        self,
        *,
        assistant_message: Msg,
        commits: RuntimeCommitPort,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        tools_enabled: bool,
        cancellation: CancellationSignal,
        interaction_projection: ToolResult | RuntimeApprovedToolCall | None = None,
    ) -> ToolBatchPlan:
        """在任何 outer permission 前识别 linked/已回答调用，保留原模型顺序。"""
        bridge = self.environment.tool_bridge
        subagent_names = {
            tool.name for tool in bridge.tool_view.active_tools if isinstance(tool, SubagentTool)
        }
        projected_id = (
            interaction_projection.tool_use_id
            if isinstance(interaction_projection, ToolResult)
            else interaction_projection.tool_call_id
            if interaction_projection is not None
            else None
        )
        continuations = {
            call.id
            for call in assistant_message.tool_calls
            if tools_enabled
            and call.name in subagent_names
            and (
                commits.load_subagent_link(tool_call_id=call.id) is not None
                or call.id == projected_id
            )
        }
        context = dict(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            workspace_root=workspace_root,
            permission_mode=permission_mode,
            metadata=metadata,
            cancellation=cancellation,
        )
        if not continuations:
            return bridge.preflight_once(
                assistant_message=assistant_message, tools_enabled=tools_enabled, **context
            )
        calls = []
        for call in assistant_message.tool_calls:
            if call.id in continuations:
                calls.append(bridge.prepare_subagent_continuation(call, **context))
            else:
                calls.extend(
                    bridge.preflight_once(
                        assistant_message=assistant_message.model_copy(update={"content": [call]}),
                        tools_enabled=tools_enabled,
                        **context,
                    ).calls
                )
        return ToolBatchPlan(calls=tuple(calls))

    async def _execute_parallel_tool_window(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        window: Sequence[tuple[int, PreparedToolCall]],
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None,
        stream_sink: RuntimeEventSink | None,
    ) -> RuntimeCursor | RuntimeActivationResult:
        """执行一个有界安全窗口，并按模型 ordinal 提交结果。"""
        prepared_calls = [prepared for _, prepared in window]
        self.environment.tool_bridge._initialize_parallel_read_state(
            activation.session_id,
            prepared_calls,
        )
        durable_guards = [
            CommitPortToolEffectGuard(
                activation=activation,
                cursor=cursor,
                commits=commits,
                workspace_root=self.environment.workspace_root,
                tool_index=tool_index,
            )
            for tool_index, _ in window
        ]
        guards = [
            (
                _LiveToolEffectGuard(
                    guard=guard,
                    sink=stream_sink,
                    activation=activation,
                    step_index=cursor.step_index,
                    tool_ordinal=tool_index + 1,
                )
                if stream_sink is not None
                else guard
            )
            for (tool_index, _), guard in zip(window, durable_guards, strict=True)
        ]
        tasks: list[asyncio.Task[_ToolCompletion]] = []
        runtime_cancelled_tasks: set[asyncio.Task[_ToolCompletion]] = set()
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
            tasks.append(asyncio.create_task(_execute_tool_with_timeout(operation, timeout)))

        pending = set(tasks)
        try:
            while pending:
                done, remaining = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if any(
                    task.cancelled() or task.exception() is not None or task.result().timed_out
                    for task in done
                ):
                    newly_cancelled = remaining - runtime_cancelled_tasks
                    runtime_cancelled_tasks.update(remaining)
                    for task in newly_cancelled:
                        task.cancel()
                pending = remaining
        except asyncio.CancelledError:
            pending = {task for task in tasks if not task.done()}
            newly_cancelled = pending - runtime_cancelled_tasks
            runtime_cancelled_tasks.update(pending)
            for task in newly_cancelled:
                task.cancel()
            # 不重复向 child 转发取消；有限 IO 必须先收回，随后按 ordinal 提交已知前缀。
            while pending:
                try:
                    _, pending = await asyncio.wait(pending)
                except asyncio.CancelledError:
                    continue

        result_slots: list[ToolResult | BaseException] = []
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
                result_slots.append(task.result().result)
                continue
            result_slots.append(exception)
            if task not in runtime_cancelled_tasks and not isinstance(
                exception, (IrisCancellationRequestedError, TimeoutError)
            ):
                infrastructure_errors.append((offset, exception))

        settlement_exception = next(
            (
                slot
                for task, slot in zip(tasks, result_slots, strict=True)
                if isinstance(slot, BaseException) and task not in runtime_cancelled_tasks
            ),
            None,
        )
        tool_timed_out = any(
            not task.cancelled() and task.exception() is None and task.result().timed_out
            for task in tasks
        )
        if settlement_exception is None and tool_timed_out:
            settlement_exception = TimeoutError()
        committed_cursor = cursor
        interrupted = False
        for offset, slot in enumerate(result_slots):
            if isinstance(slot, BaseException):
                interrupted = True
                break
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
                steering=None if tool_timed_out else steering,
                stream_sink=stream_sink,
            )

        if _task_cancellation_pending():
            raise asyncio.CancelledError
        if infrastructure_errors:
            _, infrastructure_error = min(infrastructure_errors, key=lambda item: item[0])
            raise infrastructure_error

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
        if tool_timed_out:
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.FAILED,
                cursor=committed_cursor,
                assistant_message=cursor.assistant_message,
                error=RunErrorInfo(code="TOOL_TIMEOUT", message="工具执行超时", source="tool"),
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
        stream_sink: RuntimeEventSink | None,
        subagent_call: SubagentParentCall | None = None,
    ) -> RuntimeCursor:
        """封装既有的单步 result commit 与 cursor 推进。"""
        next_index = cursor.next_tool_index + 1
        cursor_after = _project_tool_result_cursor(
            cursor,
            result,
            read_state=_read_state_snapshot(
                self.environment.tool_bridge.read_state(activation.session_id)
            ),
        )
        steering_claim = None
        if (
            steering is not None
            and subagent_call is None
            and next_index == len(cursor.tool_calls)
            and not (
                result.is_error and activation.options.tool_error_policy is ToolErrorPolicy.STOP
            )
            and not _activation_cancelled(commits, cancellation)
            and not _deadline_expired(commits)
            and not _task_cancellation_pending()
        ):
            claimed_input = await steering.claim(
                activation.run_id,
                activation.activation_id,
            )
            if claimed_input is not None:
                steering_claim = (steering, claimed_input)
        message_delta = (result.to_msg(),)
        if steering_claim is not None:
            _, claimed_input = steering_claim
            message_delta = (*message_delta, claimed_input.message)
        try:
            if subagent_call is not None:
                committed_cursor = commits.finalize_subagent_result(
                    call=subagent_call,
                    result=result,
                    cursor_after=cursor_after,
                )
            else:
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
        if stream_sink is not None:
            stream_sink.emit(
                _runtime_stream_event(
                    "tool.completed",
                    run_id=activation.run_id,
                    session_id=activation.session_id,
                    activation_id=activation.activation_id,
                    step_index=tool_call.step_index,
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    tool_ordinal=tool_call.ordinal,
                    tool_result=result,
                )
            )
        return committed_cursor

    async def _prepare_run_input(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
    ) -> RuntimeCursor | RuntimeActivationResult:
        """准备首次窗口、BCI 和用户输入，在任何模型调用前原子提交。"""
        snapshot = commits.load_session()
        if snapshot.session_id != activation.session_id:
            raise IrisRunConflictError("commit port 返回了跨 session history")
        try:
            before_current_input = self.environment.context_builder.build_before_current_input(
                self.environment.context_input.before_current_input
            )
            messages = self.environment.assembler.build_turn_messages(
                before_current_input=before_current_input,
                current_input=Msg.user(activation.run_input),
            )
            initial_window = None
            if snapshot.context_window is None:
                if self.environment.memory_service is None:
                    initial_window = SessionContextWindow()
                else:
                    pending_history = [*snapshot.messages, *messages]
                    protected = protected_message_indices(
                        pending_history, activation.initial_session_message_count
                    )
                    history = project_history(pending_history, snapshot.compaction, protected)
                    initial_window, _, _ = await self._adopt_context_window(
                        history=history,
                        options=activation.options,
                        input_budget_tokens=self.environment.agent_config.compaction.input_budget_tokens,
                    )
        except Exception as exc:
            return _failed_activation(cursor, exc)

        if _activation_cancelled(commits, cancellation):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED, cursor=cursor
            )
        if _deadline_expired(commits):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED, cursor=cursor
            )
        return commits.commit_run_input(
            RuntimeRunInputCommit(
                cursor_before=cursor,
                message_delta=tuple(messages),
                cursor_after=cursor.model_copy(update={"position": "before_model"}),
                initial_context_window=initial_window,
            )
        )

    def _build_model_request(
        self,
        *,
        history: list[Msg],
        options: RuntimeExecutionOptions,
        context_window: SessionContextWindow,
    ) -> tuple[LLMRequest, ContextBuildOutput]:
        """按同一窗口组装完整消息、模型选项和实际工具schema。"""
        context_input = self.environment.context_input.model_copy(
            update={"before_current_input": None}
        )
        context_output = self.environment.context_builder.build(
            context_input,
            system_addendum=(
                context_window.memory_overview
                if self.environment.memory_service is not None
                else ""
            ),
        )
        request = self.environment.assembler.build_request(
            agent_config=self.environment.agent_config,
            context_output=context_output,
            history=history,
            current_input=None,
        )
        request = _apply_request_options(request, options.request_options)
        request = _apply_tool_schemas(
            request,
            include_tools=options.include_tools,
            tool_view=self.environment.tool_bridge.tool_view,
            provider=self.environment.agent_config.model.provider,
        )
        return request, context_output

    async def _adopt_context_window(
        self,
        *,
        history: list[Msg],
        options: RuntimeExecutionOptions,
        input_budget_tokens: int,
    ) -> tuple[SessionContextWindow, LLMRequest, int]:
        """只在窗口采用时读取发布物并应用memory专用额度。"""
        config = self.environment.agent_config
        candidates = await load_context_windows(
            prompt_renderer=self.environment.prompt_renderer,
            memory_service=self.environment.memory_service,
            namespaces=config.memory.read_namespaces,
            tool_names=(
                [tool.name for tool in self.environment.tool_bridge.tool_view.active_tools]
                if options.include_tools
                else []
            ),
        )
        return select_context_window(
            candidates=candidates,
            build_request=lambda window: self._build_model_request(
                history=history, options=options, context_window=window
            )[0],
            provider=self.environment.provider,
            memory_budget_tokens=floor(
                config.compaction.input_budget_tokens * config.memory.overview.system_budget_ratio
            ),
            input_budget_tokens=input_budget_tokens,
        )

    async def _execute_model_step(
        self,
        *,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        steering: RuntimeSteeringPort | None,
        stream_sink: RuntimeEventSink | None,
    ) -> _ModelStepAdvance | RuntimeActivationResult:
        """执行并 required commit 一次 provider step。"""
        snapshot = commits.load_session()
        if snapshot.session_id != activation.session_id:
            raise IrisRunConflictError("commit port 返回了跨 session history")
        try:
            protected_indices = protected_message_indices(
                list(snapshot.messages), activation.initial_session_message_count
            )
            request, context_output = self._build_model_request(
                history=project_history(
                    list(snapshot.messages), snapshot.compaction, protected_indices
                ),
                options=activation.options,
                context_window=cast(SessionContextWindow, snapshot.context_window),
            )

            def build_request(history: list[Msg]) -> LLMRequest:
                messages = self.environment.assembler.build_conversation(
                    context_output=context_output,
                    history=history,
                    current_input=None,
                ).messages
                return request.model_copy(update={"messages": messages})

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
            compacted = await self._compact_request(
                request=request,
                build_request=build_request,
                snapshot=snapshot,
                protected_indices=protected_indices,
                activation=activation,
                cursor=cursor,
                commits=commits,
                cancellation=cancellation,
                stream_sink=stream_sink,
            )
        except Exception as exc:
            return _failed_activation(cursor, exc)
        if isinstance(compacted, RuntimeActivationResult):
            return compacted
        request = compacted
        # 摘要与重试已消耗原 run 的绝对 deadline，不沿用 reservation 的旧剩余额度。
        remaining = commits.remaining_deadline_seconds()
        if remaining is not None and remaining <= 0:
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED, cursor=cursor
            )
        if _activation_cancelled(commits, cancellation):
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED, cursor=cursor
            )

        if stream_sink is not None:
            stream_sink.emit(
                _runtime_stream_event(
                    "model.step.started",
                    run_id=activation.run_id,
                    session_id=activation.session_id,
                    activation_id=activation.activation_id,
                    step_index=cursor.step_index,
                )
            )

        try:
            operation = self._execute_provider_request(
                request=request,
                activation=activation,
                cursor=cursor,
                stream_sink=stream_sink,
            )
            provider_outcome = (
                await asyncio.wait_for(operation, timeout=remaining)
                if remaining is not None
                else await operation
            )
            if isinstance(provider_outcome, RuntimeActivationResult):
                return provider_outcome
            response = provider_outcome
            assistant = response.to_msg()
        except _RuntimeSinkEmissionError as exc:
            raise exc.error from exc
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
        message_delta = (assistant,)
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
                return _ModelStepAdvance(cursor=committed)

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

        _emit_tool_preparing(
            stream_sink,
            activation=activation,
            cursor=cursor,
            tool_calls=tuple(assistant.tool_calls),
            start_ordinal=1,
        )
        plan = self._prepare_tool_plan(
            assistant_message=assistant,
            commits=commits,
            session_id=activation.session_id,
            run_id=activation.run_id,
            agent_id=self.environment.agent_config.name,
            workspace_root=self.environment.workspace_root,
            permission_mode=self.environment.agent_config.permissions.writes,
            metadata={"activation_id": activation.activation_id},
            tools_enabled=activation.options.include_tools,
            cancellation=cancellation,
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
        return _ModelStepAdvance(cursor=committed, plan=plan)

    async def _compact_request(
        self,
        *,
        request: LLMRequest,
        build_request: Callable[[list[Msg]], LLMRequest],
        snapshot: SessionSnapshot,
        protected_indices: tuple[int, ...],
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
        stream_sink: RuntimeEventSink | None,
    ) -> LLMRequest | RuntimeActivationResult:
        """在同一模型步 reservation 内生成并原子安装完整摘要投影。"""
        config = self.environment.agent_config.compaction
        provider = self.environment.provider
        before = provider.estimate_input_tokens(request)
        if before < config.trigger_tokens:
            return request
        messages = list(snapshot.messages)
        end = select_compaction_end(
            messages=messages,
            previous_compaction=snapshot.compaction,
            protected_indices=protected_indices,
            config=config,
            build_request=build_request,
            estimate_input_tokens=provider.estimate_input_tokens,
        )
        if end is None:
            if before <= config.input_budget_tokens:
                return request
            raise IrisContextCompactionError(
                "输入超过预算且没有新增可压缩历史", code="CONTEXT_COMPACTION_UNAVAILABLE"
            )

        capture_port = self.environment.memory_capture_port
        if capture_port is not None:
            capture_port.request_capture(activation.run_id, len(snapshot.messages))
        loop = asyncio.get_running_loop()
        operation_deadline = loop.time() + config.timeout_seconds
        if stream_sink is not None:
            stream_sink.emit(
                _runtime_stream_event(
                    "context.compaction.started",
                    run_id=activation.run_id,
                    session_id=activation.session_id,
                    activation_id=activation.activation_id,
                    step_index=cursor.step_index,
                )
            )
        completed = False
        try:
            system_prompt = render_prompt(
                self.environment.prompt_renderer, config.prompt_path, {}
            ).strip()
            previous = snapshot.compaction
            summary = previous.summary if previous is not None else None
            start = previous.covered_message_count if previous is not None else 0
            records = serialize_history(messages[start:end], start)
            position = (0, 0)
            while position[0] < len(records):
                batch = next_summary_batch(
                    request,
                    summary,
                    records,
                    position,
                    config,
                    provider.estimate_input_tokens,
                    system_prompt=system_prompt,
                    prompt_renderer=self.environment.prompt_renderer,
                )
                for attempt in range(2):
                    stopped = _compaction_stop(cursor, commits, cancellation, operation_deadline)
                    if stopped is not None:
                        return stopped
                    timeout = operation_deadline - loop.time()
                    run_remaining = commits.remaining_deadline_seconds()
                    if run_remaining is not None:
                        timeout = min(timeout, run_remaining)
                    if batch.request.timeout is not None:
                        timeout = min(timeout, batch.request.timeout)
                    summary_request = batch.request.model_copy(update={"timeout": timeout})
                    try:
                        response = await asyncio.wait_for(
                            provider.complete(summary_request), timeout=timeout
                        )
                    except (IrisAPIConnectionError, IrisRateLimitExceededError, TimeoutError):
                        stopped = _compaction_stop(
                            cursor, commits, cancellation, operation_deadline
                        )
                        if stopped is not None:
                            return stopped
                        if attempt == 1:
                            raise
                        continue
                    break
                # 所有已返回响应（包括无效摘要）先计费；owner/fence 由提交端口裁决。
                commits.record_compaction_usage(
                    TokenUsage.model_construct(
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        total_tokens=response.total_tokens,
                    )
                )
                stopped = _compaction_stop(cursor, commits, cancellation, operation_deadline)
                if stopped is not None:
                    return stopped
                summary = consume_summary_response(response)
                position = batch.next_position

            compaction = SessionCompaction.model_construct(
                summary=summary, covered_message_count=end
            )
            history = project_history(messages, compaction, protected_indices)
            current_window = cast(SessionContextWindow, snapshot.context_window)
            if self.environment.memory_service is None and not current_window.memory_overview:
                next_window = SessionContextWindow()
                candidate = build_request(history)
                after = provider.estimate_input_tokens(candidate)
            else:
                next_window, candidate, after = await self._adopt_context_window(
                    history=history,
                    options=activation.options,
                    input_budget_tokens=config.trigger_tokens,
                )
            stopped = _compaction_stop(cursor, commits, cancellation, operation_deadline)
            if stopped is not None:
                return stopped
            if after > config.trigger_tokens or after >= before:
                raise IrisContextCompactionError(
                    "摘要后的完整请求未缩小或仍超过自动摘要额度",
                    code="CONTEXT_COMPACTION_FAILED",
                )
            commits.commit_compaction(
                RuntimeCompactionCommit(
                    cursor_before=cursor,
                    expected_session_revision=snapshot.revision,
                    compaction=compaction,
                    context_window=next_window,
                    before_input_tokens=before,
                    after_input_tokens=after,
                )
            )
            completed = True
        except IrisCancellationRequestedError:
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.CANCELLED, cursor=cursor
            )
        except Exception as exc:
            error = (
                RunErrorInfo(
                    code="PROVIDER_TIMEOUT",
                    message=str(exc) or "摘要 provider 请求超时",
                    source="provider",
                )
                if isinstance(exc, TimeoutError)
                else _normalize_run_error(exc)
            )
            if error.source == "provider":
                error = error.model_copy(
                    update={"details": {**error.details, "operation": "compaction"}}
                )
            return RuntimeActivationResult(
                outcome=RuntimeActivationOutcome.FAILED, cursor=cursor, error=error
            )
        finally:
            if not completed and stream_sink is not None:
                stream_sink.emit(
                    _runtime_stream_event(
                        "context.compaction.failed",
                        run_id=activation.run_id,
                        session_id=activation.session_id,
                        activation_id=activation.activation_id,
                        step_index=cursor.step_index,
                    )
                )
        if stream_sink is not None:
            stream_sink.emit(
                _runtime_stream_event(
                    "context.compaction.completed",
                    run_id=activation.run_id,
                    session_id=activation.session_id,
                    activation_id=activation.activation_id,
                    step_index=cursor.step_index,
                )
            )
        return candidate

    async def _execute_provider_request(
        self,
        *,
        request: LLMRequest,
        activation: RuntimeActivationInput,
        cursor: RuntimeCursor,
        stream_sink: RuntimeEventSink | None,
    ) -> LLMResponse | RuntimeActivationResult:
        """执行 complete 或 direct-pull stream，并只返回完整响应。"""
        if stream_sink is None:
            return await self.environment.provider.complete(
                request.model_copy(update={"stream": False})
            )

        provider = streaming_provider_for(self.environment.provider)
        if provider is None:
            return _failed_activation(
                cursor,
                IrisProviderStreamError(
                    "Provider 不支持 runtime streaming capability",
                    provider=self.environment.agent_config.model.provider,
                ),
            )

        stream_request = request.model_copy(update={"stream": True})
        event_stream = provider.stream(stream_request)
        try:
            async for model_event in event_stream:
                try:
                    stream_sink.emit(
                        _runtime_stream_event(
                            "model.event",
                            run_id=activation.run_id,
                            session_id=activation.session_id,
                            activation_id=activation.activation_id,
                            step_index=cursor.step_index,
                            model_event=model_event,
                        )
                    )
                except Exception as exc:
                    raise _RuntimeSinkEmissionError(exc) from exc
                if isinstance(model_event, ModelResponseCompleted):
                    return model_event.response
                if isinstance(model_event, (ModelResponseFailed, ModelResponseCancelled)):
                    return _provider_stream_failure(cursor, model_event)
        finally:
            # Terminal 会提前结束 async for，仍需释放 provider iterator 及其底层连接。
            close = getattr(event_stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    _logger.warning("关闭 provider typed stream 失败", exc_info=True)

        return _failed_activation(
            cursor,
            IrisProviderStreamInterruptedError(
                "Provider stream 在合法终态前结束",
                provider=self.environment.agent_config.model.provider,
            ),
        )

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


def _project_tool_result_cursor(
    cursor: RuntimeCursor,
    result: ToolResult,
    *,
    read_state: dict[str, Any] | None,
) -> RuntimeCursor:
    """从 pre-tool cursor 投影唯一的结果前缀与下一位置。"""
    next_index = cursor.next_tool_index + 1
    if next_index == len(cursor.tool_calls):
        return RuntimeCursor(
            position="before_model", step_index=cursor.step_index + 1, read_state=read_state
        )
    return cursor.model_copy(
        update={
            "next_tool_index": next_index,
            "tool_results": (*cursor.tool_results, result),
            "read_state": read_state,
        }
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


def _emit_tool_preparing(
    sink: RuntimeEventSink | None,
    *,
    activation: RuntimeActivationInput,
    cursor: RuntimeCursor,
    tool_calls: Sequence[ToolUseBlock],
    start_ordinal: int,
) -> None:
    """在完整 tool calls 进入 preflight 前发布 preparing facts。"""
    if sink is None:
        return
    for ordinal, tool_call in enumerate(tool_calls, start=start_ordinal):
        sink.emit(
            _runtime_stream_event(
                "tool.preparing",
                run_id=activation.run_id,
                session_id=activation.session_id,
                activation_id=activation.activation_id,
                step_index=cursor.step_index,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                tool_ordinal=ordinal,
            )
        )


def _provider_stream_failure(
    cursor: RuntimeCursor,
    event: ModelResponseFailed | ModelResponseCancelled,
) -> RuntimeActivationResult:
    """把 provider stream terminal 映射为未提交的 activation failure。"""
    error = event.error
    return RuntimeActivationResult(
        outcome=RuntimeActivationOutcome.FAILED,
        cursor=cursor,
        assistant_message=cursor.assistant_message,
        error=RunErrorInfo(
            code=error.code if error is not None else "PROVIDER_STREAM_ERROR",
            message=error.message if error is not None else "Provider stream cancelled",
            source="provider",
            details={
                "retryable": error.retryable if error is not None else False,
                "semantic_output_emitted": event.semantic_output_emitted,
                "model_stream_id": event.scope.model_stream_id,
            },
        ),
    )


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
    projection: ToolResult | RuntimeApprovedToolCall,
    subject: RuntimeToolCall,
    request: HumanInteractionRequest | None,
) -> None:
    """把 response projection 绑定到 durable cursor 当前调用，允许动态裁决不再需要 gate。"""
    if isinstance(projection, RuntimeApprovedToolCall):
        if request is not None and not isinstance(request.prompt, PermissionPrompt):
            raise IrisRunConflictError("interaction projection 类型与 question gate 不匹配")
        if (
            projection.tool_call_id != subject.tool_call_id
            or projection.tool_name != subject.tool_name
            or projection.fingerprint != subject.fingerprint
        ):
            raise IrisRunConflictError("interaction projection 与 pending gate 不匹配")
        return
    if projection.tool_use_id != subject.tool_call_id or projection.tool_name != subject.tool_name:
        raise IrisRunConflictError("interaction projection 与 pending gate 不匹配")
    if request is not None and isinstance(request.prompt, QuestionPrompt):
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


def _compaction_stop(
    cursor: RuntimeCursor,
    commits: RuntimeCommitPort,
    cancellation: CancellationSignal,
    operation_deadline: float,
) -> RuntimeActivationResult | None:
    """在摘要副作用边界检查取消和共享截止时间，保留 run 控制流语义。"""
    if _activation_cancelled(commits, cancellation):
        return RuntimeActivationResult(outcome=RuntimeActivationOutcome.CANCELLED, cursor=cursor)
    if _deadline_expired(commits):
        return RuntimeActivationResult(
            outcome=RuntimeActivationOutcome.DEADLINE_EXCEEDED, cursor=cursor
        )
    if asyncio.get_running_loop().time() >= operation_deadline:
        return _failed_activation(
            cursor,
            IrisContextCompactionError("自动摘要操作超时", code="CONTEXT_COMPACTION_TIMEOUT"),
        )
    return None


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


def _read_state_snapshot(state: ReadFileState | None) -> dict[str, Any] | None:
    """把可信 read state 序列化到 cursor。"""
    return None if state is None else state.model_dump(mode="json")


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
