"""Runtime 工具桥接。

本模块只把 assistant tool calls 连接到 `ToolExecutor` 并返回有序工具结果；它不调用
provider、不写入 session，也不决定后续 loop 行为。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..exceptions import IrisToolExecutionError
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
        self._read_states: dict[str, Any] = {}

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
        return ToolBatchPlan(calls=calls)

    def read_state(self, session_id: str) -> Any | None:
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

    async def execute_once(
        self,
        *,
        assistant_message: Msg,
        session_id: str,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        metadata: Mapping[str, Any] | None,
        tools_enabled: bool = True,
    ) -> list[ToolResult]:
        """执行助手消息中的工具调用并按原始顺序返回结果。

        Args:
            assistant_message (Msg): Provider 返回的 assistant 消息。
            session_id (str): 当前会话 ID。
            agent_id (str): 发起工具调用的 agent 标识。
            workspace_root (Path): 工具执行工作区根目录。
            permission_mode (str): 工具权限模式。
            metadata (Mapping[str, Any] | None): 运行态追踪元数据。
            tools_enabled (bool): 本轮是否允许执行工具调用。

        Returns:
            list[ToolResult]: 与 assistant tool call 顺序一致的工具结果。
        """
        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            return []

        active_names = _active_tool_names(self.tool_view) if tools_enabled else set()
        active_calls: list[ToolUseBlock] = []
        result_slots: list[ToolResult | None] = []
        for call in tool_calls:
            if call.name not in active_names:
                result_slots.append(_not_allowed_result(call))
                continue
            active_calls.append(call)
            result_slots.append(None)

        if active_calls:
            context = ToolExecutionContext(
                workspace_root=workspace_root,
                session_id=session_id,
                agent_id=agent_id,
                permission_mode=permission_mode,
                metadata=dict(metadata or {}),
                read_state=self._read_states.get(session_id),
            )
            active_results = await self.tool_executor.execute_many(active_calls, context)
            if context.read_state is not None:
                self._read_states[session_id] = context.read_state
            _merge_active_results(result_slots, active_results)

        results = [cast(ToolResult, result) for result in result_slots]
        return results


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


def _merge_active_results(
    result_slots: list[ToolResult | None],
    active_results: Sequence[ToolResult],
) -> None:
    """按原始 tool call 顺序填回执行结果。"""
    expected_count = sum(result is None for result in result_slots)
    actual_count = len(active_results)
    if actual_count != expected_count:
        raise IrisToolExecutionError(
            "工具执行结果数量不匹配",
            expected_count=expected_count,
            actual_count=actual_count,
        )

    iterator = iter(active_results)
    for index, result in enumerate(result_slots):
        if result is None:
            result_slots[index] = next(iterator)


__all__ = ["ToolBridge"]
