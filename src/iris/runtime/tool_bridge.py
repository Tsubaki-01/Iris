"""Runtime 工具桥接。

本模块只把 assistant tool calls 连接到 `ToolExecutor`，并把执行结果转换为
可回灌模型的 `Msg.tool_result()` 与 session 工具事件；它不调用 provider，也不决定
后续 loop 行为。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..exceptions import IrisToolExecutionError
from ..message import Msg, ToolUseBlock
from ..session import SessionStore
from ..tools import (
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistryView,
    ToolResult,
)
from .models import ToolBridgeResult


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

    async def execute_once(
        self,
        *,
        assistant_message: Msg,
        session_id: str,
        run_id: str,
        step_index: int,
        agent_id: str,
        workspace_root: Path,
        permission_mode: str,
        session_store: SessionStore,
        metadata: Mapping[str, Any] | None,
        tools_enabled: bool = True,
    ) -> ToolBridgeResult:
        """执行助手消息中的工具调用并追加 session 工具事件。

        Args:
            assistant_message (Msg): Provider 返回的 assistant 消息。
            session_id (str): 当前会话 ID。
            run_id (str): 当前 runtime run ID。
            step_index (int): 当前 loop 步骤序号；本阶段固定由调用方传入 0。
            agent_id (str): 发起工具调用的 agent 标识。
            workspace_root (Path): 工具执行工作区根目录。
            permission_mode (str): 工具权限模式。
            session_store (SessionStore): 工具事件写入目标。
            metadata (Mapping[str, Any] | None): 运行态追踪元数据。
            tools_enabled (bool): 本轮是否允许执行工具调用。

        Returns:
            ToolBridgeResult: 工具结果、模型回灌消息和事件快照。
        """
        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            return ToolBridgeResult()

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
            active_results = await self.tool_executor.execute_many(
                active_calls, context
            )
            if context.read_state is not None:
                self._read_states[session_id] = context.read_state
            _merge_active_results(result_slots, active_results)

        results = [cast(ToolResult, result) for result in result_slots]
        messages = [_to_tool_result_message(result) for result in results]
        events = [
            _to_tool_event(
                result,
                run_id=run_id,
                step_index=step_index,
                agent_id=agent_id,
                metadata=metadata,
            )
            for result in results
        ]
        for event in events:
            session_store.append_tool_event(session_id, event)

        return ToolBridgeResult(results=results, messages=messages, events=events)


def _active_tool_names(tool_view: ToolRegistryView) -> set[str]:
    """从活动工具视图推导本轮允许调用的工具名。"""
    return {tool.definition.name for tool in tool_view.active_tools}


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


def _to_tool_result_message(result: ToolResult) -> Msg:
    """将工具结果转换为可回灌模型的消息。"""
    return Msg.tool_result(
        tool_use_id=result.tool_use_id,
        content=result.model_content,
        is_error=result.is_error,
        name=result.tool_name,
        metadata=result.to_block_metadata(),
    )


def _to_tool_event(
    result: ToolResult,
    *,
    run_id: str,
    step_index: int,
    agent_id: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, object]:
    """构造 JSON-safe session 工具事件。"""
    error = result.error.model_dump(mode="json") if result.error is not None else None
    artifact = (
        result.artifact.model_dump(mode="json") if result.artifact is not None else None
    )
    event: dict[str, object] = {
        "type": "tool_result",
        "tool_call_id": result.tool_use_id,
        "tool_name": result.tool_name,
        "status": "error" if result.is_error else "ok",
        "error": error,
        "artifact": artifact,
        "run_id": run_id,
        "step_index": step_index,
        "agent_id": agent_id,
        "metadata": dict(metadata or {}),
    }
    return cast(dict[str, object], _json_safe(event))


def _json_safe(value: object) -> object:
    """通过 JSON round-trip 校验并清理事件里的 JSON 原生值。"""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except TypeError as exc:
        raise IrisToolExecutionError(
            "session 工具事件包含非 JSON 可序列化值",
            reason=str(exc),
        ) from exc


__all__ = ["ToolBridge"]
