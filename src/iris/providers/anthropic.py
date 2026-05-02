"""Anthropic 消息格式适配器模块。

负责整合 Iris 标准消息与 Anthropic 特定格式的相互转换过程。

Example:
    adapter = AnthropicMessageAdapter()
    api_payload = adapter.to_provider(msg)
"""

# region imports
from __future__ import annotations

import json
from typing import Any

from ..message import (
    ContentBlock,
    Msg,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .base import LLMResponse, MessageAdapter

# endregion


class AnthropicMessageAdapter(MessageAdapter):
    """序列化 Iris 消息为 Anthropic 格式，并解析其返回值。

    Anthropic 不将系统级角色包含在信息体内部，而是作为独立参数。本适配器
    主要处理助手和用户的对话流及工具调用块。

    Example:
        adapter = AnthropicMessageAdapter()
        parsed = adapter.from_provider(api_response_dict)
    """

    def to_provider(self, msg: Msg) -> dict[str, Any]:
        """将单个标准消息对象转换为 Anthropic API 支持的数据格式。

        不支持处理系统提示，系统提示应在外部独立处理并按 API 要求提交。

        Args:
            msg (Msg): Iris 的通用消息实体。

        Returns:
            dict[str, Any]: 符合 Anthropic 所需参数结构的字典。

        Raises:
            ValueError: 当输入消息包含系统角色 (System) 元素时抛出，防止被 API 拒绝。
        """
        if msg.role == Role.SYSTEM:
            raise ValueError(
                "System messages should be passed via the `system` parameter "
                "in the Anthropic API, not in the `messages` list."
            )

        api_role = "assistant" if msg.role == Role.ASSISTANT else "user"

        if isinstance(msg.content, str):
            return {
                "role": api_role,
                "content": {"type": "text", "text": msg.content},
            }

        blocks: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, ToolResultBlock):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        return {"role": api_role, "content": blocks}

    @classmethod
    def from_provider(cls, response: dict[str, Any]) -> LLMResponse:
        """从 Anthropic 原始 API 的响应报文构造出一个标准的 LLM 响应对象。

        处理文本回复并展开可能有的工具请求，统一收集用量数据。

        Args:
            response (dict[str, Any]): API 请求返回的 JSON 字典。

        Returns:
            LLMResponse: 转换后的 LLM 响应对象。
        """
        # --- 1. 初始化收集器 ---
        blocks: list[ContentBlock] = []

        # --- 2. 迭代解析各内容块 ---
        for raw_block in response.get("content", []):
            block_type = raw_block.get("type")
            if block_type == "text":
                blocks.append(TextBlock(text=raw_block["text"]))
            elif block_type == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=raw_block["id"],
                        name=raw_block["name"],
                        input=_loads_dict(raw_block.get("input", {})),
                    )
                )

        # --- 3. 提取元信息并组装 ---
        usage = response.get("usage", {})
        return LLMResponse(
            provider="anthropic",
            id=response.get("id", ""),
            model=response.get("model", ""),
            finish_reason=response.get("stop_reason", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            content=blocks,
            metadata={"stop_reason": response.get("stop_reason", "")},
        )


def _loads_dict(value: Any) -> dict[str, Any]:
    """尝试将可能为 JSON 字符串的值安全转换为字典。

    用于保障由各方传入的可选或不规范字典字段稳定可用。

    Args:
        value (Any): 被转型的对象，大多应为字典或序列化字符。

    Returns:
        dict[str, Any]: 若存在问题可妥善退化返回空字典。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}
