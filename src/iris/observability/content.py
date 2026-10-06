"""将模型可见内容投影为固定 GenAI 格式，不读取文件或遍历业务元数据。

格式依据 semantic-conventions-genai 的 e07f4ebacb08f56db8c4c882d117720333fbca04。
调用方在进入本模块前统一检查采集策略及 span.is_recording，并隔离投影异常。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry.util.types import AttributeValue

from ..message import ContentBlock, ImageBlock, LLMRequest, LLMResponse, TextBlock, ToolUseBlock

if TYPE_CHECKING:
    from ..tools.base import ToolResult


def request_attributes(request: LLMRequest, max_chars: int) -> dict[str, AttributeValue]:
    """记录 effective 请求的消息与工具定义。

    系统消息已经属于 Iris 请求历史，按规范保留在 input.messages 中；不重复
    投影为仅适用于独立指令的 system_instructions。
    """
    values: dict[str, Any] = {
        "gen_ai.input.messages": [
            {"role": message.role.value, "parts": _parts(message.content)}
            for message in request.messages
        ]
    }
    if request.tools:
        values["gen_ai.tool.definitions"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in request.tools
        ]
    return _serialize(values, max_chars)


def response_attributes(response: LLMResponse, max_chars: int) -> dict[str, AttributeValue]:
    """记录 typed 响应正文及已有 reasoning；结束原因由 span 属性表达。"""
    parts = _parts(response.content)
    if response.reasoning:
        parts.insert(0, {"type": "reasoning", "content": response.reasoning})
    return _serialize(
        {"gen_ai.output.messages": [{"role": "assistant", "parts": parts}]}, max_chars
    )


def tool_attributes(
    arguments: dict[str, Any], result: ToolResult | None, max_chars: int
) -> dict[str, AttributeValue]:
    """记录实际参数与最终模型结果，保留错误替换、图片和 Hook 反馈。"""
    values: dict[str, Any] = {"gen_ai.tool.call.arguments": arguments}
    if result is not None:
        values["gen_ai.tool.call.result"] = {"parts": _parts(result.model_blocks)}
    return _serialize(values, max_chars)


def _parts(content: str | Sequence[ContentBlock]) -> list[dict[str, Any]]:
    """直接投影可信内容块；图片仅使用已保存的模型副本 URI 与 MIME。"""
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            part: dict[str, Any] = {"type": "text", "content": block.text}
        elif isinstance(block, ImageBlock):
            part = {
                "type": "uri",
                "modality": "image",
                "uri": block.model.path.as_uri(),
                "mime_type": block.model.mime_type,
            }
        elif isinstance(block, ToolUseBlock):
            part = {
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.input,
            }
        else:
            part = {
                "type": "tool_call_response",
                "id": block.tool_use_id,
                "response": {"parts": _parts(block.content)},
            }
        parts.append(part)
    return parts


def _serialize(values: dict[str, Any], max_chars: int) -> dict[str, AttributeValue]:
    """超限内容只保留文本预览，不将不完整 JSON 发布到标准属性。"""
    attributes: dict[str, AttributeValue] = {}
    truncated: list[str] = []
    for key, value in values.items():
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(encoded) <= max_chars:
            attributes[key] = encoded
        else:
            attributes[f"iris.content.{key}.preview"] = encoded[:max_chars]
            truncated.append(key)
    if truncated:
        attributes["iris.content.truncated_fields"] = truncated
    return attributes
