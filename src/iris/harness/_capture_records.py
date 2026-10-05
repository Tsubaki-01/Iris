"""已提交原文的共享事实投影，不依赖 Memory 或项目经验的材料模型。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..lifecycle.history import RunMessageSlice
from ..message import Role, TextBlock, ToolResultBlock, ToolUseBlock


@dataclass(frozen=True, slots=True)
class CapturedRecord:
    """捕获队列内部的一条事实，领域消费者分别组装自己的持久记录。"""

    ref: str
    role: str
    text: str
    occurred_at: str
    message_ordinal: int
    block_index: int
    tool_event: bool
    metadata: dict[str, Any]


def capture_records(messages: RunMessageSlice, *, end: int) -> tuple[CapturedRecord, ...]:
    """保留绝对消息/块引用，Memory 与 Skill 读回只保留引用和后续反馈。"""
    records: list[CapturedRecord] = []
    for ordinal, message in enumerate(messages.messages, messages.start_message_count):
        if ordinal >= end:
            break
        if message.role is Role.SYSTEM or message.metadata.get("context_kind") in {
            "before_current_input",
            "memory",
        }:
            continue
        occurred_at = datetime.fromtimestamp(message.timestamp, UTC).isoformat()
        for block_index, block in enumerate(message.blocks):
            record_id = f"{messages.source_id}:{messages.run_id}:{ordinal}:{block_index}"
            metadata: dict[str, Any] = {"message_ordinal": ordinal, "block_index": block_index}
            tool_event = False
            if isinstance(block, TextBlock):
                text, role = block.text, message.role.value
            elif isinstance(block, ToolUseBlock):
                role, text = "assistant", json.dumps(block.input, ensure_ascii=False)
                metadata.update(tool_name=block.name, call_id=block.id, record_kind="tool_call")
                if block.name in {"memory_search", "memory_fetch", "load_skill"}:
                    metadata.update(evidence_allowed=False, query=block.input)
                    text = ""
                if block.name in {"memory_update", "memory_forget", "memory_fetch"}:
                    item_id = block.input.get("item_id")
                    if isinstance(item_id, str):
                        metadata["memory_item_ids"] = [item_id]
            elif isinstance(block, ToolResultBlock):
                role, tool_event = "tool", True
                metadata.update(
                    tool_name=block.name,
                    call_id=block.tool_use_id,
                    is_error=block.is_error,
                    record_kind="tool_result",
                )
                if block.name in {"memory_search", "memory_fetch", "load_skill"}:
                    metadata["evidence_allowed"] = False
                    text = ""
                    if block.name != "load_skill":
                        metadata["memory_item_ids"] = _memory_item_ids(block.text)
                else:
                    text = block.text
                    if block.name in {"memory_remember", "memory_update"}:
                        metadata["memory_item_ids"] = _memory_item_ids(block.text)
                artifact = block.metadata.get("artifact")
                if artifact is not None:
                    metadata["artifact"] = artifact
            else:
                continue
            records.append(
                CapturedRecord(
                    record_id,
                    role,
                    text,
                    occurred_at,
                    ordinal,
                    block_index,
                    tool_event,
                    metadata,
                )
            )
    return tuple(records)


def _memory_item_ids(content: str) -> list[str]:
    """在工具原始 JSON 边界只提取 ID，读回正文不进入新事实。"""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    items = payload.get("items", [payload.get("item")])
    if not isinstance(items, list):
        return []
    return [
        item_id
        for item in items
        if isinstance(item, dict)
        if isinstance(item_id := item.get("item_id", item.get("id")), str)
    ]
