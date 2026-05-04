"""Anthropic 消息格式适配器。

Anthropic Messages API 将 system prompt 放在顶层 `system` 字段，并将工具
结果放入 user 消息的 `tool_result` 内容块。本模块把这些差异限制在 adapter
内部，避免上层业务直接依赖 Anthropic payload。

Example:
    >>> from iris.message import LLMRequest, Msg
    >>> request = LLMRequest(
    ...     model="claude-sonnet-4-5",
    ...     messages=[Msg.system("规则"), Msg.user("你好")],
    ... )
    >>> AnthropicMessageAdapter().to_provider_request(request)["system"]
    '规则'
"""

# region imports
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..message import Msg, Role, TextBlock, ToolResultBlock, ToolUseBlock
from ..message.llm import LLMRequest, LLMResponse
from .adapter import ProviderAdapter

# endregion


class AnthropicMessageAdapter(ProviderAdapter):
    """Anthropic Messages API 格式适配器。

    该类只做 Anthropic JSON payload 与 Iris 内部模型之间的转换。HTTP 调用、
    鉴权 header 和错误映射由 `ProviderClient` 负责。

    Attributes:
        provider (str): 固定为 `"anthropic"`。

    Example:
        >>> AnthropicMessageAdapter().provider
        'anthropic'
    """

    provider: str = "anthropic"

    def to_provider_request(self, request: LLMRequest) -> dict[str, Any]:
        """将 Iris 请求转换为 Anthropic payload。

        Args:
            request (LLMRequest): Provider-neutral 的一次模型调用请求。

        Returns:
            dict[str, Any]: Anthropic Messages API 请求 payload。

        Example:
            >>> request = LLMRequest(model="claude", messages=[Msg.user("你好")])
            >>> AnthropicMessageAdapter().to_provider_request(request)["messages"][0]["role"]
            'user'
        """
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self.format_messages(request.non_system_messages()),
        }
        system_prompt = request.system_prompt()
        if system_prompt:
            payload["system"] = system_prompt
        self._append_common_options(payload, request)
        return payload

    def from_provider_response(self, response: Mapping[str, Any]) -> LLMResponse:
        """将 Anthropic raw response 转换为标准响应。

        Args:
            response (Mapping[str, Any]): Anthropic API 返回的原始 JSON 对象。

        Returns:
            LLMResponse: 标准化后的 Iris 响应模型。

        Example:
            >>> raw = {"id": "x", "model": "claude", "content": [{"type": "text", "text": "hi"}]}
            >>> AnthropicMessageAdapter().from_provider_response(raw).to_msg().text
            'hi'
        """
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return LLMResponse(
            provider=self.provider,
            id=str(response.get("id") or ""),
            model=str(response.get("model") or ""),
            content=self._content_blocks(response.get("content") or []),
            finish_reason=str(response.get("stop_reason") or ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            metadata={"type": response.get("type")} if response.get("type") else {},
        )

    def format_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        """转换消息列表为 Anthropic 消息格式。

        Args:
            messages (list[Msg]): 已排除 system prompt 的 Iris 消息列表。

        Returns:
            list[dict[str, Any]]: Anthropic Messages API 的 messages 列表。

        Example:
            >>> AnthropicMessageAdapter().format_messages([Msg.user("你好")])[0]["role"]
            'user'
        """
        return [self._format_message(msg) for msg in messages]

    def _format_message(self, msg: Msg) -> dict[str, Any]:
        """转换单条 Iris 消息为 Anthropic message。"""
        role = "assistant" if msg.role == Role.ASSISTANT else "user"
        return {"role": role, "content": self._format_content(msg)}

    def _format_content(self, msg: Msg) -> list[dict[str, Any]]:
        """转换 Iris 内容块为 Anthropic content block。"""
        content: list[dict[str, Any]] = []
        for block in msg.blocks:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, ToolResultBlock):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        return content

    def _append_common_options(self, payload: dict[str, Any], request: LLMRequest) -> None:
        """追加 Anthropic Messages API 支持的通用请求选项。"""
        option_names = ("temperature", "top_p", "max_tokens", "tool_choice")
        for name in option_names:
            value = getattr(request, name)
            if value is not None:
                payload[name] = value
        if request.tools:
            payload["tools"] = request.tools

    def _content_blocks(self, content: list[Any]) -> list[TextBlock | ToolUseBlock]:
        """从 Anthropic content block 中提取 Iris 内容块。"""
        blocks: list[TextBlock | ToolUseBlock] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text":
                blocks.append(TextBlock(text=str(item.get("text") or "")))
            elif item.get("type") == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=str(item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        input=dict(item.get("input") or {}),
                    )
                )
        return blocks
