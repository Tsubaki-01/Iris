"""OpenAI 消息格式适配器。

默认生成 Chat Completions payload，因为 Iris 的 `Conversation.messages`
与 Chat Completions 的 messages 输入天然对应。Responses API 作为显式
`provider_options["api_style"] = "responses"` 路径保留在同一个 adapter 内。

Example:
    >>> from iris.message import LLMRequest, Msg
    >>> adapter = OpenAIMessageAdapter()
    >>> adapter.to_provider_request(LLMRequest(model="gpt-4o", messages=[Msg.user("你好")]))
    {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': '你好', 'name': 'user'}]}
"""

# region imports
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from ..exceptions import IrisValidationError
from ..message import ContentBlock, Msg, Role, TextBlock, ToolResultBlock, ToolUseBlock
from ..message.llm import LLMRequest, LLMResponse
from .adapter import ProviderAdapter

# endregion


class OpenAIMessageAdapter(ProviderAdapter):
    """OpenAI 消息格式适配器，默认使用 Chat Completions。

    该类只负责 Iris 内部模型与 OpenAI JSON payload 之间的转换，不发起
    HTTP 请求，也不处理 API key。网络职责由 `ProviderClient` 负责。

    Attributes:
        provider (str): 固定为 `"openai"`。
        default_api_style (str): 默认 API 风格，固定为 `"chat"`。

    Example:
        >>> adapter = OpenAIMessageAdapter()
        >>> adapter.default_api_style
        'chat'
    """

    provider: str = "openai"
    default_api_style: str = "chat"

    def to_provider_request(self, request: LLMRequest) -> dict[str, Any]:
        """将 Iris 请求转换为 OpenAI payload。

        Args:
            request (LLMRequest): Provider-neutral 的一次模型调用请求。

        Returns:
            dict[str, Any]: OpenAI Chat Completions 或 Responses 请求 payload。

        Raises:
            IrisValidationError: `provider_options["api_style"]` 不是支持值时抛出。

        Example:
            >>> request = LLMRequest(model="gpt-4o", messages=[Msg.user("你好")])
            >>> OpenAIMessageAdapter().to_provider_request(request)["messages"][0]["role"]
            'user'
        """
        api_style = self._api_style(request)
        payload: dict[str, Any] = {"model": request.model}
        formatted_messages = self.format_messages(request.messages, api_style=api_style)
        if api_style == "chat":
            payload["messages"] = formatted_messages
        else:
            payload["input"] = formatted_messages
        self._append_common_options(payload, request)
        return payload

    def from_provider_response(self, response: Mapping[str, Any]) -> LLMResponse:
        """将 OpenAI raw response 转换为标准响应。

        Args:
            response (Mapping[str, Any]): OpenAI API 返回的原始 JSON 对象。

        Returns:
            LLMResponse: 标准化后的 Iris 响应模型。

        Example:
            >>> raw = {"id": "x", "model": "gpt-4o", "choices": [{"message": {"content": "hi"}}]}
            >>> OpenAIMessageAdapter().from_provider_response(raw).to_msg().text
            'hi'
        """
        if "choices" in response:
            return self._from_chat_response(response)
        return self._from_responses_response(response)

    def format_messages(
        self,
        messages: list[Msg],
        *,
        api_style: Literal["chat", "responses"] = "chat",
    ) -> list[dict[str, Any]]:
        """转换消息列表为 OpenAI 消息格式。

        Args:
            messages (list[Msg]): Iris 内部消息列表。

        Returns:
            list[dict[str, Any]]: OpenAI messages/input 可接收的消息列表。

        Example:
            >>> OpenAIMessageAdapter().format_messages([Msg.user("你好")])
            [{'role': 'user', 'content': '你好', 'name': 'user'}]
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.extend(self._format_message(msg, api_style=api_style))
        return result

    def _api_style(self, request: LLMRequest) -> Literal["chat", "responses"]:
        """解析 OpenAI API 风格。"""
        api_style = request.provider_options.get("api_style", self.default_api_style)
        if api_style not in {"chat", "responses"}:
            raise IrisValidationError("不支持的 OpenAI API 风格", api_style=api_style)
        return api_style

    def _format_message(
        self,
        msg: Msg,
        *,
        api_style: Literal["chat", "responses"],
    ) -> list[dict[str, Any]]:
        """转换单条 Iris 消息为 OpenAI 消息。"""
        if msg.tool_results:
            return [
                self._format_tool_result(block, api_style=api_style) for block in msg.tool_results
            ]

        item: dict[str, Any] = {"role": msg.role, "content": msg.text}
        if msg.sender and msg.role == Role.USER:
            item["name"] = msg.sender
        if msg.tool_calls:
            item["tool_calls"] = [self._format_tool_call(block) for block in msg.tool_calls]
        return [item]

    def _format_tool_result(
        self,
        block: ToolResultBlock,
        *,
        api_style: Literal["chat", "responses"],
    ) -> dict[str, Any]:
        """转换工具结果块为 OpenAI tool message。"""
        if api_style == "responses":
            return {
                "type": "function_call_output",
                "call_id": block.tool_use_id,
                "output": block.content,
            }
        return {
            "role": "tool",
            "tool_call_id": block.tool_use_id,
            "content": block.content,
        }

    def _format_tool_call(self, block: ToolUseBlock) -> dict[str, Any]:
        """转换工具调用块为 OpenAI function tool call。"""
        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": json.dumps(block.input, ensure_ascii=False, separators=(",", ":")),
            },
        }

    def _append_common_options(self, payload: dict[str, Any], request: LLMRequest) -> None:
        """追加 OpenAI Chat 与 Responses 共享的请求选项。"""
        option_names = (
            "temperature",
            "top_p",
            "max_tokens",
            "tool_choice",
            "response_format",
        )
        for name in option_names:
            value = getattr(request, name)
            if value is not None:
                payload[name] = value
        if request.tools:
            payload["tools"] = request.tools

    def _from_chat_response(self, response: Mapping[str, Any]) -> LLMResponse:
        """解析 OpenAI Chat Completions 响应。"""
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        content_blocks = self._content_blocks_from_chat_message(message)
        usage = response.get("usage") or {}
        return LLMResponse(
            provider=self.provider,
            id=str(response.get("id") or ""),
            model=str(response.get("model") or ""),
            content=content_blocks,
            finish_reason=str(choice.get("finish_reason") or ""),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            metadata=({"raw_object": response.get("object")} if response.get("object") else {}),
        )

    def _from_responses_response(self, response: Mapping[str, Any]) -> LLMResponse:
        """解析 OpenAI Responses 响应。"""
        usage = response.get("usage") or {}
        return LLMResponse(
            provider=self.provider,
            id=str(response.get("id") or ""),
            model=str(response.get("model") or ""),
            content=self._content_blocks_from_responses_output(response.get("output") or []),
            finish_reason=str(response.get("status") or ""),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            metadata=({"raw_object": response.get("object")} if response.get("object") else {}),
        )

    def _content_blocks_from_chat_message(
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

    def _content_blocks_from_responses_output(
        self,
        output: list[Any],
    ) -> list[ContentBlock]:
        """从 Responses output 中提取 Iris 内容块。"""
        blocks: list[ContentBlock] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "function_call":
                blocks.append(
                    ToolUseBlock(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        input=self._parse_arguments(item.get("arguments")),
                    )
                )
            for content in item.get("content") or []:
                if isinstance(content, Mapping) and content.get("type") in {
                    "output_text",
                    "text",
                }:
                    blocks.append(TextBlock(text=str(content.get("text") or "")))
        return blocks

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        """解析 OpenAI 工具调用参数。"""
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str) or not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
