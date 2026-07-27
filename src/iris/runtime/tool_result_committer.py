"""Runtime 工具结果的统一 session 提交。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..message import Msg
from ..session import SessionStore
from ..tools import ToolResult
from .models import ToolResultCommit
from .tool_results import build_tool_result_event, build_tool_result_message


def commit_tool_results(
    *,
    results: Sequence[ToolResult],
    session_store: SessionStore,
    session_id: str,
    run_id: str,
    step_index: int,
    agent_id: str,
    metadata: Mapping[str, Any] | None,
    deduplicate_messages: bool,
) -> ToolResultCommit:
    """投影工具结果并写入 session message 与 event。

    Args:
        results: 当前批次按工具调用顺序排列的执行结果。
        session_store: 当前 session 的持久化目标。
        session_id: 目标 session 标识。
        run_id: 生成 event ID 的 runtime run 标识。
        step_index: 当前 loop 步骤序号。
        agent_id: 产生工具调用的 agent 标识。
        metadata: event 追踪字段。
        deduplicate_messages: 是否跳过 history 中已存在的工具调用 message。

    Returns:
        ToolResultCommit: 原始结果及其消息、事件投影。
    """
    result_list = list(results)
    if not result_list:
        return ToolResultCommit()

    messages = project_tool_result_messages(result_list)
    events = [
        build_tool_result_event(
            result,
            run_id=run_id,
            step_index=step_index,
            agent_id=agent_id,
            metadata=metadata,
        )
        for result in result_list
    ]
    history = [Msg.from_dict(item) for item in session_store.load_messages(session_id)]
    existing_ids = {block.tool_use_id for message in history for block in message.tool_results}
    messages_to_append = [
        message
        for result, message in zip(result_list, messages, strict=True)
        if not deduplicate_messages or result.tool_use_id not in existing_ids
    ]
    if messages_to_append:
        history.extend(messages_to_append)
        session_store.save_messages(
            session_id,
            [message.model_dump(mode="json") for message in history],
        )
    for event in events:
        session_store.append_tool_event(session_id, event)
    return ToolResultCommit(results=result_list, messages=messages, events=events)


def project_tool_result_messages(results: Sequence[ToolResult]) -> list[Msg]:
    """将工具结果按原始顺序投影为 provider-neutral 消息。"""
    return [build_tool_result_message(result) for result in results]


__all__ = ["commit_tool_results", "project_tool_result_messages"]
