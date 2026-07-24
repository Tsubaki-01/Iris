"""Runtime 工具结果的消息与事件投影。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ..exceptions import IrisToolExecutionError
from ..message import Msg
from ..tools import ToolResult


def build_tool_result_message(result: ToolResult) -> Msg:
    """将结构化工具结果转换为 provider-neutral 消息。"""
    return Msg.tool_result(
        tool_use_id=result.tool_use_id,
        content=result.model_content,
        is_error=result.is_error,
        name=result.tool_name,
        metadata=result.to_block_metadata(),
    )


def build_tool_result_event(
    result: ToolResult,
    *,
    run_id: str,
    step_index: int,
    agent_id: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, object]:
    """构造包含稳定 ID 的 JSON-safe session 工具事件。"""
    error = result.error.model_dump(mode="json") if result.error is not None else None
    artifact = result.artifact.model_dump(mode="json") if result.artifact is not None else None
    event: dict[str, object] = {
        "event_id": f"tool_result:{run_id}:{result.tool_use_id}",
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


__all__ = ["build_tool_result_event", "build_tool_result_message"]
