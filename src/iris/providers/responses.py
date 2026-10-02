"""原生 Responses 消息投影、响应解析与本地计量输入。"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from copy import deepcopy
from typing import Any, cast

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
from ._encoding import chat_tool_choice, chat_tools, function_schema


class ResponsesMapper:
    """保持 typed 正文权威，仅用 metadata 重放原生 item 顺序。"""

    def format_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        """将有效历史转换为完整 Responses input，不引入服务端会话状态。"""
        items: list[dict[str, Any]] = []
        for message in messages:
            replay = message.metadata.get("responses")
            if message.role is Role.ASSISTANT and replay is not None:
                items.extend(self._replay_assistant(message, replay["items"]))
                continue
            text_parts: list[dict[str, str]] = []
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    text_parts.append({"type": "input_text", "text": block.text})
                    continue
                if text_parts:
                    items.append(self._message_item(message.role, text_parts))
                    text_parts = []
                if isinstance(block, ToolUseBlock):
                    items.append(self._tool_call(block))
                elif isinstance(block, ToolResultBlock):
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": [{"type": "input_text", "text": block.content}],
                        }
                    )
            if text_parts:
                items.append(self._message_item(message.role, text_parts))
        return items

    def parse_response(self, data: Mapping[str, Any], *, provider: str) -> LLMResponse:
        """解析完整终态；失败保留已知 usage，不交付可执行的部分工具调用。"""
        usage = data.get("usage") or {}
        tokens = {
            name: int(usage.get(name) or 0)
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        status = data.get("status")
        error_context: dict[str, Any] = {"provider": provider, "status": status}
        if data.get("usage") is not None:
            error_context["usage"] = tokens
        if status != "completed":
            reason = data.get("error") or data.get("incomplete_details") or status
            raise IrisProviderError(
                f"Responses 响应未完成: {status}", reason=reason, **error_context
            )

        blocks: list[ContentBlock] = []
        replay: list[dict[str, Any]] = []
        reasoning: list[str] = []
        for item in data.get("output") or []:
            kind = item["type"]
            identity: dict[str, Any] = {
                key: item[key] for key in ("id", "phase", "status") if item.get(key) is not None
            }
            if kind == "message":
                indices = []
                annotations = []
                for part in item.get("content") or []:
                    if part["type"] == "refusal":
                        raise IrisProviderError(
                            "Responses 拒绝生成回答", reason=part["refusal"], **error_context
                        )
                    if part["type"] == "output_text":
                        annotations.append(deepcopy(part.get("annotations", [])))
                        indices.append(len(blocks))
                        blocks.append(TextBlock(text=part["text"]))
                replay.append(
                    {"type": kind, **identity, "block_indices": indices, "annotations": annotations}
                )
            elif kind == "function_call":
                try:
                    arguments = json.loads(item["arguments"])
                except (json.JSONDecodeError, TypeError) as exc:
                    raise IrisProviderError(
                        "Responses 工具参数不是合法 JSON object", **error_context
                    ) from exc
                if (
                    not isinstance(arguments, dict)
                    or not item.get("call_id")
                    or not item.get("name")
                ):
                    raise IrisProviderError("Responses 工具调用缺少有效参数或身份", **error_context)
                replay.append({"type": kind, **identity, "block_index": len(blocks)})
                blocks.append(ToolUseBlock(id=item["call_id"], name=item["name"], input=arguments))
            elif kind == "reasoning":
                replay.append({"type": kind, "item": deepcopy(item)})
                reasoning.extend(self._reasoning_text(item))

        return LLMResponse(
            provider=provider,
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            content=blocks,
            finish_reason="tool_calls"
            if any(isinstance(b, ToolUseBlock) for b in blocks)
            else "stop",
            **tokens,
            reasoning="\n".join(reasoning),
            metadata={"responses": {"status": status, "items": replay}},
        )

    def _replay_assistant(
        self, message: Msg, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        blocks = message.blocks
        items: list[dict[str, Any]] = []
        for record in records:
            kind = record["type"]
            if kind == "reasoning":
                items.append(deepcopy(record["item"]))
                continue
            identity: dict[str, Any] = {
                key: record[key] for key in ("id", "phase", "status") if key in record
            }
            if kind == "message":
                content = [
                    {
                        "type": "output_text",
                        "text": cast(TextBlock, blocks[index]).text,
                        "annotations": annotations,
                    }
                    for index, annotations in zip(
                        record["block_indices"], record["annotations"], strict=True
                    )
                ]
                items.append({**self._message_item(Role.ASSISTANT, content), **identity})
            elif kind == "function_call":
                block = cast(ToolUseBlock, blocks[record["block_index"]])
                items.append({**self._tool_call(block), **identity})
        return items

    @staticmethod
    def _message_item(role: Role, content: list[dict[str, str]]) -> dict[str, Any]:
        return {"type": "message", "role": role.value, "content": content}

    @staticmethod
    def _tool_call(block: ToolUseBlock) -> dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": block.id,
            "name": block.name,
            "arguments": json.dumps(block.input, ensure_ascii=False, separators=(",", ":")),
        }

    @staticmethod
    def _reasoning_text(item: Mapping[str, Any]) -> list[str]:
        return [
            part["text"]
            for field in ("summary", "content")
            for part in item.get(field) or []
            if isinstance(part.get("text"), str)
        ]

    def token_count_messages(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从实际 input 提取本地 tokenizer 形状；此投影不用于发送。"""
        messages: list[dict[str, Any]] = []
        for item in items:
            kind = item["type"]
            if kind == "message":
                messages.append(
                    {"role": item["role"], "content": "\n".join(p["text"] for p in item["content"])}
                )
            elif kind == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": item["call_id"],
                                "function": {"name": item["name"], "arguments": item["arguments"]},
                            }
                        ],
                    }
                )
            elif kind == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": "\n".join(part["text"] for part in item["output"]),
                    }
                )
            elif kind == "reasoning":
                text = "\n".join(self._reasoning_text(item))
                if text:
                    messages.append({"role": "assistant", "content": text})
        return messages


def responses_text_format(response_format: ResponseFormat) -> dict[str, Any]:
    """将逻辑输出模式编码为 Responses text.format。"""
    if isinstance(response_format, str):
        return {"type": response_format}
    return {"type": "json_schema", **response_format}


class ResponsesAdapter:
    """固定的原生 Responses 编码、调用和计量适配。"""

    def encode_request(self, request: LLMRequest, *, transport: str) -> dict[str, Any]:
        """生成协议参数；拒绝尚未接入的原生传输。"""
        if transport != "openai":
            raise IrisProviderError(
                "尚未接入该 provider 的原生 Responses 传输", litellm_provider=transport
            )
        kwargs: dict[str, Any] = {
            "input": ResponsesMapper().format_messages(request.messages),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if request.tools:
            kwargs["tools"] = [
                {"type": "function", **function_schema(tool)} for tool in request.tools
            ]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = (
                {"type": "function", **request.tool_choice}
                if isinstance(request.tool_choice, dict)
                else request.tool_choice
            )
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["text"] = {"format": responses_text_format(request.response_format)}
        if "reasoning_effort" in request.provider_options:
            kwargs["reasoning"] = {"effort": request.provider_options["reasoning_effort"]}
        return kwargs

    async def invoke(self, kwargs: dict[str, Any]) -> Any:
        """通过 LiteLLM 调用 Responses，沿用其模型流式支持策略。"""
        response = await litellm.aresponses(**kwargs)
        return _closing_response_stream(response) if kwargs.get("stream") else response

    def parse_response(self, data: Mapping[str, Any], *, provider: str) -> LLMResponse:
        """以同一 mapper 解析 complete 和最终 streaming response。"""
        return ResponsesMapper().parse_response(data, provider=provider)

    def token_count_projection(self, request: LLMRequest, *, transport: str) -> dict[str, Any]:
        """从实际 input 提取 tokenizer 的 Chat 形状，不计 opaque reasoning。"""
        kwargs = self.encode_request(request, transport=transport)
        return {
            "messages": ResponsesMapper().token_count_messages(kwargs["input"]),
            "tools": chat_tools(request.tools),
            "tool_choice": chat_tool_choice(request.tool_choice),
        }

    def format_for_count(self, response_format: ResponseFormat) -> dict[str, Any]:
        """返回实际结构化输出格式，供本地计量。"""
        return responses_text_format(response_format)


async def _closing_response_stream(raw_stream: Any) -> AsyncGenerator[Any, None]:
    """结束消费时关闭本次 HTTP 响应，连接池仍由 LiteLLM 管理。"""
    try:
        async for event in raw_stream:
            yield event
    finally:
        await raw_stream.response.aclose()
