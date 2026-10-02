"""Chat Completions 协议适配。

将 Iris 逻辑请求编码为 Chat Completions 参数，通过 LiteLLM 调用，
并统一解析响应、重放推理字段和投影本地 token 计量输入。
"""

# region imports
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import litellm

from ..exceptions import IrisProviderError
from ..message import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Msg,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from ..message.llm import ResponseFormat
from ._tool_encoding import chat_tool_choice, chat_tools

# endregion


class ChatCompletionsMapper:
    """在 Iris 消息与 Chat Completions 消息格式之间转换。"""

    def format_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        """转换消息列表为 Chat Completions messages 形状。"""
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.extend(self._format_message(msg))
        return result

    def _format_message(self, msg: Msg) -> list[dict[str, Any]]:
        """转换单条 Iris 消息为 Chat Completions message。"""
        if msg.tool_results:
            return [self._format_tool_result(block) for block in msg.tool_results]

        item: dict[str, Any] = {"role": msg.role, "content": msg.text}
        replay = msg.metadata.get("chat_completions")
        if msg.role is Role.ASSISTANT and replay is not None:
            item[replay["reasoning_field"]] = msg.metadata.get("reasoning", "")
        if msg.sender and msg.role == Role.USER:
            item["name"] = msg.sender
        if msg.tool_calls:
            item["tool_calls"] = [self._format_tool_call(block) for block in msg.tool_calls]
        return [item]

    def _format_tool_result(self, block: ToolResultBlock) -> dict[str, Any]:
        """转换工具结果块为 Chat Completions tool message。"""
        return {
            "role": "tool",
            "tool_call_id": block.tool_use_id,
            "content": block.text,
        }

    def _format_tool_call(self, block: ToolUseBlock) -> dict[str, Any]:
        """转换工具调用块为 Chat Completions function tool call。"""
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
        """在 provider 边界解析完整工具参数，不交付无效工具。"""
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise IrisProviderError("Chat 工具参数不是合法 JSON object") from exc
        if not isinstance(parsed, dict):
            raise IrisProviderError("Chat 工具参数必须是 JSON object")
        return parsed

    def parse_response(self, data: Mapping[str, Any], *, provider: str) -> LLMResponse:
        """归一化成功终态，失败保留用量且不返回工具意图。"""
        usage = data.get("usage") or {}
        tokens = {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        context: dict[str, Any] = {"provider": provider}
        if data.get("usage") is not None:
            context["usage"] = tokens
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise IrisProviderError("Chat 响应必须包含一个 choice", **context)
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "tool_calls"}:
            raise IrisProviderError("Chat 响应未完成", status=finish_reason, **context)
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise IrisProviderError("Chat 拒绝生成回答", **context)
        try:
            blocks = self.content_blocks_from_chat_message(message)
            for block in blocks:
                if isinstance(block, ToolUseBlock) and (not block.id or not block.name):
                    raise IrisProviderError("Chat 工具调用缺少有效身份")
        except (IrisProviderError, KeyError, TypeError, ValueError) as exc:
            raise IrisProviderError("Chat 响应内容无效", **context) from exc
        reasoning_field = chat_reasoning_field(message)
        metadata = {"raw_object": data.get("object") or "chat.completion"}
        if reasoning_field is not None:
            metadata["chat_completions"] = {"reasoning_field": reasoning_field}
        return LLMResponse(
            provider=provider,
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            content=blocks,
            finish_reason="tool_calls"
            if any(isinstance(block, ToolUseBlock) for block in blocks)
            else "stop",
            reasoning=message[reasoning_field] if reasoning_field is not None else "",
            metadata=metadata,
            **tokens,
        )


def chat_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    """将逻辑输出模式编码为 Chat response_format。"""
    if isinstance(response_format, str):
        return {"type": response_format}
    return {"type": "json_schema", "json_schema": dict(response_format)}


class ChatCompletionsAdapter:
    """固定的 Chat Completions 编码、调用和计量适配。"""

    def encode_request(self, request: LLMRequest, *, transport: str) -> dict[str, Any]:
        """生成 Chat Completions 参数；传输 provider 路由由 client 交给 LiteLLM。"""
        kwargs: dict[str, Any] = {
            "messages": ChatCompletionsMapper().format_messages(request.messages)
        }
        if request.tools:
            kwargs["tools"] = chat_tools(request.tools)
        if request.tool_choice is not None:
            kwargs["tool_choice"] = chat_tool_choice(request.tool_choice)
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = chat_response_format(request.response_format)
        if "reasoning_effort" in request.provider_options:
            kwargs["reasoning_effort"] = request.provider_options["reasoning_effort"]
        if request.stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def invoke(self, kwargs: dict[str, Any]) -> Any:
        """通过 LiteLLM 调用 Chat Completions，协议在 client 构造时已确定。"""
        return await litellm.acompletion(**kwargs)

    def parse_response(self, data: Mapping[str, Any], *, provider: str) -> LLMResponse:
        """解析完整 Chat 响应。"""
        return ChatCompletionsMapper().parse_response(data, provider=provider)

    def token_count_projection(self, request: LLMRequest, *, transport: str) -> dict[str, Any]:
        """使用实际 Chat messages、工具和选择生成本地计量输入。"""
        kwargs = self.encode_request(request, transport=transport)
        return {
            "messages": kwargs["messages"],
            "tools": kwargs.get("tools", []),
            "tool_choice": kwargs.get("tool_choice"),
        }

    def format_for_count(self, response_format: ResponseFormat) -> dict[str, Any]:
        """返回实际结构化输出格式，供本地计量。"""
        return chat_response_format(response_format)


def chat_reasoning_field(message: Mapping[str, Any]) -> str | None:
    """标识 Chat 原生 reasoning 字段，只存来源不复制推理正文。"""
    return next(
        (key for key in ("reasoning_content", "reasoning") if message.get(key) is not None), None
    )
