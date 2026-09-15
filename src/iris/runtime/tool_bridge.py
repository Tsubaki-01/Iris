"""Runtime 工具桥接。

本模块只把 assistant tool calls 连接到 `ToolExecutor` 并返回有序工具结果；它不调用
provider、不写入 session，也不决定后续 loop 行为。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..message import Msg, ToolUseBlock
from ..tools import (
    CancellationSignal,
    PreparedToolCall,
    ReadFileState,
    ToolBatchPlan,
    ToolEffectGuard,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistryView,
    ToolResult,
)
from ..tools.subagent import SubagentExecutionOutcome, SubagentParentCall


class ToolBridge:
    """执行一次 assistant tool call bridge。"""

    def __init__(
        self,
        *,
        tool_view: ToolRegistryView,
        tool_executor: ToolExecutor,
    ) -> None:
        """创建工具桥接器。

        Args:
            tool_view (ToolRegistryView): 本轮暴露给模型的工具视图。
            tool_executor (ToolExecutor): 实际执行工具调用的统一入口。
        """
        self.tool_view = tool_view
        self.tool_executor = tool_executor
        self._read_states: dict[str, ReadFileState] = {}

    def preflight_once(
        self,
        *,
        assistant_message: Msg,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        tools_enabled: bool = True,
        cancellation: CancellationSignal | None = None,
    ) -> ToolBatchPlan:
        """无副作用预检当前 assistant 消息中的所有活动工具调用。"""
        active_names = _active_tool_names(self.tool_view) if tools_enabled else set()
        context = self._execution_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            workspace_root=workspace_root,
            permission_mode=permission_mode,
            metadata=metadata,
            cancellation=cancellation,
        )
        active_calls = [call for call in assistant_message.tool_calls if call.name in active_names]
        prepared = iter(self.tool_executor.prepare_many(active_calls, context).calls)
        calls: list[PreparedToolCall] = []
        for call in assistant_message.tool_calls:
            if call.name in active_names:
                calls.append(next(prepared))
            else:
                calls.append(
                    PreparedToolCall(
                        tool_use=call,
                        preflight_result=_not_allowed_result(call),
                    )
                )
        return ToolBatchPlan(calls=tuple(calls))

    def read_state(self, session_id: str) -> ReadFileState | None:
        """返回 session 当前保存的文件读取状态。"""
        return self._read_states.get(session_id)

    def restore_read_state(self, session_id: str, state: dict[str, Any] | None) -> None:
        """从 checkpoint 恢复 session 的文件读取状态。"""
        if state is None:
            self._read_states.pop(session_id, None)
            return
        self._read_states[session_id] = ReadFileState.model_validate(state)

    def _is_parallel_candidate(self, prepared: PreparedToolCall) -> bool:
        """委托 executor 判断预检调用能否进入并发窗口。"""
        return self.tool_executor._is_read_only_concurrency_safe(prepared)

    def _initialize_parallel_read_state(
        self,
        session_id: str,
        prepared_calls: Sequence[PreparedToolCall],
    ) -> None:
        """在并发 child 创建前初始化唯一的文件读取状态。"""
        if self._read_states.get(session_id) is not None:
            return
        if any(
            prepared.tool is not None and prepared.tool.definition.group == "file"
            for prepared in prepared_calls
        ):
            self._read_states[session_id] = ReadFileState()

    async def execute_prepared(
        self,
        prepared: PreparedToolCall,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        cancellation: CancellationSignal,
        effect_guard: ToolEffectGuard,
        approved_tool_call_id: str | None = None,
    ) -> ToolResult:
        """用 shared signal 与 required effect guard 执行一条预检调用。"""
        context = self._execution_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            workspace_root=workspace_root,
            permission_mode=permission_mode,
            metadata=metadata,
            cancellation=cancellation,
        )
        result = await self.tool_executor.execute_prepared(
            prepared,
            context,
            approved_tool_call_id=approved_tool_call_id,
            effect_guard=effect_guard,
        )
        if context.read_state is not None:
            self._read_states[session_id] = context.read_state
        return result

    def prepare_subagent_continuation(
        self,
        tool_use: ToolUseBlock,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        cancellation: CancellationSignal | None = None,
    ) -> PreparedToolCall:
        """委托 raw-only continuation prepare；link 选择由 lifecycle caller 决定。"""
        context = self._execution_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            workspace_root=workspace_root,
            permission_mode=permission_mode,
            metadata=metadata,
            cancellation=cancellation,
        )
        return self.tool_executor.prepare_subagent_continuation(tool_use, context)

    async def execute_subagent_prepared(
        self,
        prepared: PreparedToolCall,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        cancellation: CancellationSignal | None = None,
        approved_tool_call_id: str | None = None,
        linked_continuation: bool = False,
    ) -> SubagentExecutionOutcome:
        """只在此构造 parent identity，然后驱动专用 child 执行入口。"""
        context = self._execution_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            workspace_root=workspace_root,
            permission_mode=permission_mode,
            metadata=metadata,
            cancellation=cancellation,
        )
        return await self.tool_executor.execute_subagent_prepared(
            prepared,
            context,
            parent_call=SubagentParentCall(run_id, prepared.tool_use.id),
            approved_tool_call_id=approved_tool_call_id,
            linked_continuation=linked_continuation,
        )

    def _normalize_subagent_result(
        self,
        tool_use: ToolUseBlock,
        result: ToolResult,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """WAITING child 完成后只归一化 parent identity/artifact，不重新执行或鉴权。"""
        context = self._execution_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            workspace_root=workspace_root,
            permission_mode=permission_mode,
            metadata=metadata,
            cancellation=None,
        )
        return self.tool_executor._finalize_result(
            tool_use=tool_use,
            tool=self.tool_view.get(tool_use.name),
            result=result,
            context=context,
        )

    def _execution_context(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        cancellation: CancellationSignal | None,
    ) -> ToolExecutionContext:
        """构造复用同一 read state 与 cancellation 的工具上下文。"""
        return ToolExecutionContext(
            workspace_root=workspace_root,
            session_id=session_id,
            agent_id=agent_id,
            permission_mode=permission_mode,
            metadata={**dict(metadata or {}), "run_id": run_id},
            read_state=self._read_states.get(session_id),
            cancellation=cancellation,
        )


def _active_tool_names(tool_view: ToolRegistryView) -> set[str]:
    """从活动工具视图推导本轮允许调用的工具名（含别名）。"""
    names: set[str] = set()
    for tool in tool_view.active_tools:
        names.add(tool.definition.name)
        names.update(tool.definition.aliases)
    return names


def _not_allowed_result(call: ToolUseBlock) -> ToolResult:
    """构造未暴露工具的错误结果。"""
    return ToolResult(
        tool_use_id=call.id,
        tool_name=call.name,
        is_error=True,
        error=ToolErrorInfo(
            code="TOOL_NOT_ALLOWED",
            message=f"工具未暴露给当前模型: {call.name}",
        ),
    )


__all__ = ["ToolBridge"]
