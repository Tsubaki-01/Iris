"""OpenAI 消息格式适配器模块。

负责 Iris 消息体至 OpenAI API 请求体，及该平台特定响应到 Iris 消息的转换，并支持多种风格如响应层或原生聊天体。

Example:
    adapter = OpenAIMessageAdapter(api_style="chat")
    payload = adapter.to_provider(msg)
"""

# region imports
from __future__ import annotations

import json
from typing import Any, Literal

from ..message import (
    ContentBlock,
    Msg,
    Role,
    TextBlock,
    ToolUseBlock,
)
from .base import LLMResponse, MessageAdapter

# endregion


class OpenAIMessageAdapter(MessageAdapter):
    """序列化并解析针对基于 OpenAI API 草案设计下的大模型交互消息。

    该类可以依据接口调用的偏好 (`api_style`) 处理两种模式：`responses` 与 `chat`。
    支持至少 3 种及以上的转换步骤及功能封装（故利用较重的区域拆分）。

    Attributes:
        api_style (str): 适配 OpenAI 提供商风格，可选值为 "responses" 或 "chat" 字典模式。

    Example:
        adapter = OpenAIMessageAdapter(api_style="chat")
        parsed = adapter.from_provider(api_return_dict)
    """

    api_style: Literal["responses", "chat"] = "chat"

    # ==========================================
    #               API 路由分发
    # ==========================================
    # region
    def to_provider(self, msg: Msg) -> list[dict[str, Any]]:
        """顶级路由分发入口，基于预设层风格选取合适的转换动作。

        Args:
            msg (Msg): Iris 的业务层面通用消息块。

        Returns:
            list[dict[str, Any]]: 具体特定提供商所需构建列表。
        """
        if self.api_style == "chat":
            return self.to_chat(msg)
        return self.to_responses(msg)

    def from_provider(self, response: dict[str, Any]) -> LLMResponse:
        """顶级路由捕获入口，通过设定类型从 OpenAI 返回构建 LLMResponse。

        Args:
            response (dict[str, Any]): API 实际响应结果负载。

        Returns:
            LLMResponse: 对象化标准载体。
        """
        if self.api_style == "chat":
            return self.from_chat(response)
        return self.from_responses(response)

    # endregion

    # ==========================================
    #               Chat Completions 风格API
    # ==========================================
    # region Chat Completions
    def to_chat(self, msg: Msg) -> list[dict[str, Any]]:
        """使用常见标准大模型对话列表模式 (Chat Completions) 转录信息。

        Args:
            msg (Msg): 基础消息载体。

        Returns:
            list[dict]: 转化为 role-content/tool_calls 列表。
        """
        # --- 1. 纯文本系统提示 ---
        if msg.role == Role.SYSTEM:
            return [{"role": "system", "content": msg.text}]

        # --- 2. 不带工具调用的用户纯信息 ---
        if msg.role == Role.USER and not msg.tool_results:
            return [{"role": "user", "content": msg.text}]

        # --- 3. 工具调用信息 ---
        if msg.role == Role.ASSISTANT:
            converted: dict[str, Any] = {
                "role": "assistant",
                "content": msg.text or None,
            }
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": _json_dumps(tool_call.input),
                        },
                    }
                    for tool_call in msg.tool_calls
                ]
            return [converted]

        # --- 4. 工具回调结果 ---
        if msg.tool_results:
            return [
                {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_use_id,
                    "content": tool_result.content,
                }
                for tool_result in msg.tool_results
            ]

        # 默认回退逻辑。
        return [{"role": "user", "content": msg.text}]

    @classmethod
    def from_chat(cls, response: dict[str, Any]) -> LLMResponse:
        """分解常规对话端点输出中包含的角色反馈。

        Args:
            response (dict): 大模型响应内容。

        Returns:
            LLMResponse: 格式化完毕的模型回复类实例。
        """
        choices = response.get("choices", [])
        if not choices:
            return LLMResponse(
                provider="openai",
                id=response.get("id", ""),
                model=response.get("model", ""),
            )

        choice = choices[0]
        message = choice.get("message", {})
        blocks: list[ContentBlock] = []

        text = message.get("content")
        if text:
            blocks.append(TextBlock(text=text))

        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            args = function.get("arguments", tool_call.get("arguments", {}))
            blocks.append(
                ToolUseBlock(
                    id=tool_call["id"],
                    name=function.get("name", ""),
                    input=_loads_dict(args),
                )
            )

        usage = response.get("usage", {})
        return LLMResponse(
            provider="openai",
            id=response.get("id", ""),
            model=response.get("model", ""),
            finish_reason=choice.get("finish_reason", ""),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            content=blocks,
        )

    # endregion

    # ==========================================
    #               Responses 风格API
    # ==========================================
    # region Responses
    def to_responses(self, msg: Msg) -> list[dict[str, Any]]:
        """提供针对 OpenAI responses 定义格式的处理支持。

        Args:
            msg (Msg): 构建来源实体。

        Returns:
            list[dict]: 承接具体工具调用或回应输出的数据数组。
        """
        items: list[dict[str, Any]] = []

        if msg.role == Role.SYSTEM:
            items.append({"role": "system", "content": msg.text})
            return items

        if msg.role == Role.USER and not msg.tool_results:
            items.append({"role": "user", "content": msg.text})
            return items

        if msg.role == Role.ASSISTANT:
            if msg.text:
                items.append({"role": "assistant", "content": msg.text})
            for tool_call in msg.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": _json_dumps(tool_call.input),
                    }
                )
            return items

        if msg.tool_results:
            for tool_result in msg.tool_results:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_result.tool_use_id,
                        "output": tool_result.content,
                    }
                )
            return items

        items.append({"role": "user", "content": msg.text})
        return items

    @classmethod
    def from_responses(cls, response: dict[str, Any]) -> LLMResponse:
        """从特殊 response API 输出序列中过滤提取有效节点块。

        需兼顾对 output_text 以及 function_call 体的提取及理由推断摘要。

        Args:
            response (dict): 带有嵌套 output 的原始负载数据字典。

        Returns:
            LLMResponse: 封装后具有附带元分析资料的结果信息体。
        """
        # --- 1. 获取顶层属性与初筛载体 ---
        output_list: list[dict[str, Any]] = response.get("output", [])
        blocks: list[ContentBlock] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"

        # --- 2. 依规解析输出序列体 ---
        for item in output_list:
            item_type = item.get("type")

            if item_type == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "output_text":
                        blocks.append(TextBlock(text=content_part["text"]))
            elif item_type == "function_call":
                blocks.append(
                    ToolUseBlock(
                        id=item.get("call_id", item["id"]),
                        name=item.get("name", ""),
                        input=_loads_dict(item.get("arguments", {})),
                    )
                )
            elif item_type == "reasoning":
                for summary_part in item.get("summary", []):
                    if summary_part.get("type") == "summary_text":
                        reasoning_parts.append(summary_part["text"])

        # --- 3. 确立终止标识并将信息整合打包 ---
        if output_list and output_list[-1].get("type") == "function_call":
            finish_reason = "function_call"

        usage = response.get("usage", {})
        return LLMResponse(
            provider="openai",
            id=response.get("id", ""),
            model=response.get("model", ""),
            finish_reason=finish_reason,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            reasoning="\n".join(reasoning_parts),
            content=blocks,
        )

    # endregion


# ==========================================
#               Helper Functions
# ==========================================
# region
def _json_dumps(obj: Any) -> str:
    """提供统一的非转义紧凑型 JSON 序列化功能，常用于 API 请求压缩。

    Args:
        obj (Any): 支持被格式转换的标准字典负载或序列。

    Returns:
        str: 无多余空格且不对中文字符进行 ASCII 转化的精炼字符串。
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _loads_dict(value: Any) -> dict[str, Any]:
    """尝试将未知类型的来源数据提取成强类型的字典并处理出错退行逻辑。

    Args:
        value (Any): 被传入进行转换的可能含错 JSON 或者已被预处理的内容。

    Returns:
        dict[str, Any]: 通过过滤清洗与容差判断后的字典。
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


# endregion
