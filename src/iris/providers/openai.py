"""OpenAI Chat 消息格式 helper。

本模块只保留 LiteLLM Chat Completion active path 需要的 OpenAI Chat
消息格式化与响应解析逻辑，不再提供公开 adapter API。
"""

# region imports
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..message import ContentBlock, Msg, Role, TextBlock, ToolResultBlock, ToolUseBlock

# endregion


class OpenAIChatMapper:
    """OpenAI Chat 消息格式 helper。"""

    def format_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        """转换消息列表为 OpenAI Chat messages 形状。"""
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.extend(self._format_message(msg))
        return result

    def _format_message(self, msg: Msg) -> list[dict[str, Any]]:
        """转换单条 Iris 消息为 OpenAI Chat message。"""
        if msg.tool_results:
            return [self._format_tool_result(block) for block in msg.tool_results]

        item: dict[str, Any] = {"role": msg.role, "content": msg.text}
        if msg.sender and msg.role == Role.USER:
            item["name"] = msg.sender
        if msg.tool_calls:
            item["tool_calls"] = [self._format_tool_call(block) for block in msg.tool_calls]
        return [item]

    def _format_tool_result(self, block: ToolResultBlock) -> dict[str, Any]:
        """转换工具结果块为 OpenAI Chat tool message。"""
        return {
            "role": "tool",
            "tool_call_id": block.tool_use_id,
            "content": block.content,
        }

    def _format_tool_call(self, block: ToolUseBlock) -> dict[str, Any]:
        """转换工具调用块为 OpenAI Chat function tool call。"""
        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": json.dumps(block.input, ensure_ascii=False, separators=(",", ":")),
            },
        }

    def content_blocks_from_chat_message(
        self,
        message: Mapping[str, Any],
    ) -> list[ContentBlock]:
        """从 Chat Completions message 中提取 Iris 内容块。"""
        blocks: list[ContentBlock] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            blocks.append(TextBlock(text=content))
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            blocks.append(
                ToolUseBlock(
                    id=str(tool_call.get("id") or ""),
                    name=str(function.get("name") or ""),
                    input=self._parse_arguments(function.get("arguments")),
                )
            )
        return blocks

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        """解析 OpenAI Chat 工具调用参数。"""
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str) or not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
