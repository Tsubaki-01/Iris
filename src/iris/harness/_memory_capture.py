"""将已提交会话原文投影成不可变记忆材料，保留来源而不重复学习读回记忆。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..lifecycle import RunRecord
from ..lifecycle.history import RunMessageSlice
from ..memory.generation_models import MemoryCaptureSource
from ..memory.models import MemoryEpisode, MemoryRecord, MemorySourceType
from ..message import Role, TextBlock, ToolResultBlock, ToolUseBlock


def capture_source(run: RunRecord, *, source_id: str, namespace: str) -> MemoryCaptureSource:
    """以 run admission 时的原文截点构造首次登记，不纳入继承历史。"""
    return MemoryCaptureSource(
        lifecycle_source_id=source_id,
        run_id=run.run_id,
        session_id=run.session_id,
        namespace=namespace,
        initial_message_count=run.initial_session_message_count,
        captured_until=run.initial_session_message_count,
    )


def capture_episode(
    source: MemoryCaptureSource, messages: RunMessageSlice, *, through_count: int | None = None
) -> tuple[MemoryCaptureSource, MemoryEpisode | None]:
    """只投影选中已提交后缀，终态封口不越过调用方提示的范围。"""
    end = messages.end_message_count
    if through_count is not None:
        end = max(messages.start_message_count, min(end, through_count))
    records: list[MemoryRecord] = []
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
            source_type = MemorySourceType.MESSAGE
            if isinstance(block, TextBlock):
                text = block.text
                role = message.role.value
            elif isinstance(block, ToolUseBlock):
                role = "assistant"
                text = json.dumps(block.input, ensure_ascii=False)
                metadata.update(tool_name=block.name, call_id=block.id, record_kind="tool_call")
                if block.name in {"memory_search", "memory_fetch"}:
                    metadata.update(evidence_allowed=False, query=block.input)
                    text = ""
                if block.name in {"memory_update", "memory_forget", "memory_fetch"}:
                    item_id = block.input.get("item_id")
                    if isinstance(item_id, str):
                        metadata["memory_item_ids"] = [item_id]
            elif isinstance(block, ToolResultBlock):
                role = "tool"
                source_type = MemorySourceType.TOOL_EVENT
                metadata.update(
                    tool_name=block.name,
                    call_id=block.tool_use_id,
                    is_error=block.is_error,
                    record_kind="tool_result",
                )
                if block.name in {"memory_search", "memory_fetch"}:
                    metadata["memory_item_ids"] = _memory_item_ids(block.content)
                    metadata["evidence_allowed"] = False
                    text = ""
                else:
                    text = block.content
                    if block.name in {"memory_remember", "memory_update"}:
                        metadata["memory_item_ids"] = _memory_item_ids(block.content)
                artifact = block.metadata.get("artifact")
                if artifact is not None:
                    metadata["artifact"] = artifact
            else:
                continue
            records.append(
                MemoryRecord(
                    id=record_id,
                    role=role,
                    text=text,
                    source_type=source_type,
                    source_id=record_id,
                    occurred_at=occurred_at,
                    metadata=metadata,
                )
            )
    terminal = messages.terminal_message_count if end == messages.end_message_count else None
    updated = source.model_copy(
        update={
            "captured_until": end,
            "terminal_message_count": terminal,
            "outcome": messages.outcome.value
            if terminal is not None and messages.outcome
            else None,
        }
    )
    episode = None
    if records:
        episode = MemoryEpisode(
            namespace=source.namespace,
            source_type=MemorySourceType.TASK,
            source_id=source.run_id,
            records=tuple(records),
            metadata={
                "lifecycle_source_id": source.lifecycle_source_id,
                "session_id": source.session_id,
                "start_message_count": messages.start_message_count,
                "end_message_count": end,
                "outcome": updated.outcome,
            },
        )
    return updated, episode


def _memory_item_ids(content: str) -> list[str]:
    """在工具原始 JSON 边界只提取 ID，任何读回正文均不进入新证据。"""
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
