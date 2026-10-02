"""两种协议共用的函数字段与本地 tokenizer 投影。"""

from __future__ import annotations

from typing import Any

from ..message.llm import ToolChoice, ToolSpec


def function_schema(tool: ToolSpec) -> dict[str, Any]:
    """把逻辑工具字段映射为两种 API 共用的 function 内容。"""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": tool.strict,
    }


def chat_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """生成 Chat 工具包装，同时供本地 tokenizer 使用。"""
    return [{"type": "function", "function": function_schema(tool)} for tool in tools]


def chat_tool_choice(choice: ToolChoice | None) -> str | dict[str, Any] | None:
    """生成 Chat 强制工具选择；命名策略保持原值。"""
    if isinstance(choice, dict):
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice
