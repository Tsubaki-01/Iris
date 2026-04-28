"""消息系统。

为 agent、LLM 与工具之间的通信提供统一消息类型。
所有消息都通过同一个 `Msg` 类型流转：用户输入、LLM 回复、
工具调用和工具结果都以统一结构表示。

Example:
    conv = Conversation()
    conv.add(Msg.user("请总结最近两条消息"))
"""

# region imports
from __future__ import annotations

import json
import time
import uuid
from curses import raw
from enum import Enum
from typing import Any, Literal, Self, Sequence

from pydantic import BaseModel, Field, model_validator

# endregion

# region definitions
# ==========================================
#                 枚举定义
# ==========================================


class Role(str, Enum):
    """消息发送方角色，与常见 LLM API 约定保持一致。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# ==========================================
#                 内容块定义
# ==========================================


class TextBlock(BaseModel):
    """用户或助手发送的纯文本内容。"""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """由 LLM 发起的工具调用请求。

    Attributes:
        id: 由 LLM 分配的唯一标识（用于关联结果）。
        name: 已注册的工具名（如 "bash"、"read_file"）。
        input: 工具参数，需可被 JSON 序列化。
    """

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:12]}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """工具执行后的返回结果。

    Attributes:
        tool_use_id: 对应 `ToolUseBlock` 的 `id`。
        content: 工具执行产生的文本输出。
        is_error: 工具执行是否失败。
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str = ""
    is_error: bool = False


# agent 支持的所有内容块联合类型。
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

# endregion

# region Core
# ==========================================
#                 核心消息
# ==========================================


class Msg(BaseModel):
    """通用消息单元。

    每一次交互（用户输入、LLM 回复、工具调用、工具结果）
    都表示为一个 `Msg`。这样可在系统内保持内存、序列化与
    API 格式的一致性。

    Attributes:
        role: 发送方角色。
        content: 纯字符串或内容块列表。
        sender: 可选发送方名称（多 agent 场景）。
        timestamp: 消息创建时的 Unix 时间戳（秒）。
        metadata: 任意键值元数据（如链路追踪、成本统计）。

    Examples:
        >>> user_msg = Msg.user("修复 main.py 里的 bug")
        >>> user_msg.role
        <Role.USER: 'user'>
        >>> user_msg.text
        '修复 main.py 里的 bug'

        >>> tool_result_msg = Msg.tool_result(tool_use_id="tool_abc", content="OK", is_error=False)
        >>> tool_result_msg.role
        <Role.USER: 'user'>
    """

    role: Role
    content: str | list[ContentBlock] = ""
    sender: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}

    # ==========================================
    #                 数据校验
    # ==========================================
    @model_validator(mode="after")
    def _normalize_content(self) -> Self:
        """确保内部内容格式可被统一处理。"""
        if isinstance(self.content, str):
            # 保留原始字符串，在属性访问或 to_* 序列化时再按需转换。
            pass
        return self

    # ==========================================
    #                 便捷属性
    # ==========================================
    @property
    def text(self) -> str:
        """提取并拼接消息中的文本内容。

        Returns:
            所有 `TextBlock` 文本的拼接结果，或原始字符串。
        """
        if isinstance(self.content, str):
            return self.content
        parts = [b.text for b in self.content if isinstance(b, TextBlock)]
        return "\n".join(parts)

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        """提取助手消息中的所有工具调用块。"""
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        """提取所有工具结果块（主要用于 role=tool 的消息）。"""
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ToolResultBlock)]

    @property
    def has_tool_calls(self) -> bool:
        """判断当前消息是否包含工具调用。"""
        return len(self.tool_calls) > 0

    # ==========================================
    #               内容块辅助属性
    # ==========================================
    @property
    def blocks(self) -> list[ContentBlock]:
        """始终以内容块列表形式返回消息内容。"""
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)] if self.content else []
        return list(self.content)

    # ==========================================
    #                 工厂方法
    # ==========================================
    @classmethod
    def system(cls, content: str, **kwargs: Any) -> Msg:
        """创建系统提示消息。"""
        return cls(role=Role.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, *, sender: str = "user", **kwargs: Any) -> Msg:
        """创建用户消息。"""
        return cls(role=Role.USER, content=content, sender=sender, **kwargs)

    @classmethod
    def assistant(
        cls,
        content: str | list[ContentBlock],
        *,
        sender: str = "assistant",
        **kwargs: Any,
    ) -> Msg:
        """创建助手消息。"""
        return cls(role=Role.ASSISTANT, content=content, sender=sender, **kwargs)

    @classmethod
    def tool_result(
        cls,
        *,
        tool_use_id: str,
        content: str = "",
        is_error: bool = False,
        **kwargs: Any,
    ) -> Msg:
        """创建返回给 LLM 的工具结果消息。

        Anthropic API 要求将工具结果作为 user 角色消息，
        并放在 `tool_result` 内容块中。
        """
        block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )
        return cls(role=Role.USER, content=[block], **kwargs)

    # ==========================================
    #                OpenAI API Format
    # ==========================================
    def to_openai(self, api_style: str = "responses") -> list[dict[str, Any]]:
        if api_style == "responses":
            return self.to_openai_responses()
        else:
            return self.to_openai_chat()

    def to_openai_chat(self) -> list[dict[str, Any]]:
        """转换为 OpenAI Chat Completions API 格式。

        OpenAI 的结构与 Anthropic 不同：
        - `tool_use` 会变成助手消息上的 `tool_calls`
        - `tool_result` 需要拆成 role="tool" 的独立消息

        Returns:
            返回字典列表。
        """
        if self.role == Role.SYSTEM:
            return [{"role": "system", "content": self.text}]

        if self.role == Role.USER and not self.tool_results:
            return [{"role": "user", "content": self.text}]

        # 助手消息可同时包含文本和工具调用。
        if self.role == Role.ASSISTANT:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": self.text or None,
            }
            if self.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _json_dumps(tc.input),
                        },
                    }
                    for tc in self.tool_calls
                ]
            return [msg]

        # 工具结果在 OpenAI 中需要拆分为多条 role=tool 消息。
        if self.tool_results:
            return [
                {
                    "role": "tool",
                    "tool_call_id": tr.tool_use_id,
                    "content": tr.content,
                }
                for tr in self.tool_results
            ]

        return [{"role": "user", "content": self.text}]

    def to_openai_responses(self) -> list[dict[str, Any]]:
        """转换为 OpenAI Responses API 的 input 项格式。

        Returns:
            一个或多个输入项字典的列表，可直接拼接到请求的 `input` 数组里。
        """
        items: list[dict[str, Any]] = []

        if self.role == Role.SYSTEM:
            items.append({"role": "system", "content": self.text})
            return items

        if self.role == Role.USER and not self.tool_results:
            items.append({"role": "user", "content": self.text})
            return items

        # 助手消息：文本和工具调用分别生成独立的 input item
        if self.role == Role.ASSISTANT:
            if self.text:
                items.append({"role": "assistant", "content": self.text})
            for tc in self.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": _json_dumps(tc.input),
                    }
                )
            return items

        # 工具结果：每个结果生成一个 function_call_output 项
        if self.tool_results:
            for tr in self.tool_results:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tr.tool_use_id,
                        "output": tr.content,
                    }
                )
            return items

        # 兜底：普通用户消息（理论上不会走到这里）
        items.append({"role": "user", "content": self.text})
        return items

    # ==========================================
    #              Anthropic API Format
    # ==========================================
    def to_anthropic(self) -> dict[str, Any]:
        """转换为 Anthropic Messages API 格式。

        Returns:
            可直接追加到 Anthropic 请求 `messages` 列表的字典。

        Raises:
            ValueError: 当角色为 SYSTEM 时抛出。
                系统消息应放在顶层 `system` 参数，而非 `messages`。
        """
        if self.role == Role.SYSTEM:
            raise ValueError(
                "System messages should be passed via the `system` parameter "
                "in the Anthropic API, not in the `messages` list."
            )

        api_role = "assistant" if self.role == Role.ASSISTANT else "user"

        # 字符串内容直接透传。
        if isinstance(self.content, str):
            return {"role": api_role, "content": {"type": "text", "text": self.content}}

        # 内容块按类型转换为 API 结构。
        blocks: list[dict[str, Any]] = []
        for block in self.content:
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

    # ==========================================
    #               响应反序列化
    # ==========================================
    def from_openai(
        self, response: dict[str, Any], *, api_style: str = "responses"
    ) -> Msg:
        if api_style == "responses":
            return self.from_openai_responses(response)
        else:
            return self.from_openai_chat(response)

    @classmethod
    def from_openai_chat(cls, response: dict[str, Any]) -> Msg:
        """将 OpenAI Chat Completion 响应解析为 `Msg`。

        Args:
            response: OpenAI API 返回的原始 JSON 字典。
                期望包含 `choices`（至少一个），可选 `model`、`usage`。

        Returns:
            已正确映射内容块类型的助手消息。
        """
        choices = response.get("choices", [])
        if not choices:
            # 无选择时返回空助手消息（可按需处理）
            return cls.assistant(content=[], metadata={})

        # 通常只取第一个 choice
        choice = choices[0]
        message = choice.get("message", {})

        blocks: list[ContentBlock] = []

        # 文本内容
        text = message.get("content")
        if text:
            blocks.append(TextBlock(text=text))

        # 工具调用
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            blocks.append(
                ToolUseBlock(
                    id=tc["id"],
                    name=fn.get("name", ""),
                    input=args,
                )
            )

        # 提取用量与元数据
        usage = response.get("usage", {})
        metadata = {
            "model": response.get("model", ""),
            "finish_reason": choice.get("finish_reason", ""),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        return cls.assistant(content=blocks, metadata=metadata)

    @classmethod
    def from_openai_responses(cls, response: dict[str, Any]) -> Msg:
        """将 OpenAI Responses API 的响应解析为 `Msg`。

        Args:
            response: Responses API 返回的原始 JSON 字典。
                期望顶层包含 `output` 列表，可选 `model`、`usage`。

        Returns:
            已映射内容块类型的助手消息。
        """
        output_list: list[dict] = response.get("output", [])
        blocks: list[ContentBlock] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"

        for item in output_list:
            item_type = item.get("type")

            # 1. 助手文本消息
            if item_type == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "output_text":
                        blocks.append(TextBlock(text=content_part["text"]))

            # 2. 工具调用
            elif item_type == "function_call":
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                blocks.append(
                    ToolUseBlock(
                        id=item["id"],
                        name=item.get("name", ""),
                        input=args,
                    )
                )

            # 3. 推理过程
            elif item_type == "reasoning":
                for summary_part in item.get("summary", []):
                    if summary_part.get("type") == "summary_text":
                        reasoning_parts.append(summary_part["text"])

            # 其他未知类型可忽略或记录日志

        # 推断 finish_reason
        if output_list:
            last_item = output_list[-1]
            if last_item.get("type") == "function_call":
                finish_reason = "function_call"
            # TODO 可扩展：如果最后一个 message 中有 refusal 等

        usage = response.get("usage", {})
        metadata = {
            "model": response.get("model", ""),
            "finish_reason": finish_reason,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if reasoning_parts:
            metadata["reasoning"] = "\n".join(reasoning_parts)

        return cls.assistant(content=blocks, metadata=metadata)

    @classmethod
    def from_anthropic(cls, response: dict[str, Any]) -> Msg:
        """将 Anthropic API 响应解析为 `Msg`。

        Args:
            response: Anthropic API 返回的原始 JSON 字典。
                期望包含 `role`、`content`（内容块列表），可选 `usage`。

        Returns:
            已正确映射内容块类型的助手消息。
        """
        blocks: list[ContentBlock] = []
        for raw_block in response.get("content", []):
            btype = raw_block.get("type")
            if btype == "text":
                blocks.append(TextBlock(text=raw_block["text"]))
            elif btype == "tool_use":
                args = raw_block.get("input", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                blocks.append(
                    ToolUseBlock(
                        id=raw_block["id"],
                        name=raw_block["name"],
                        input=args,
                    )
                )
            # 如 thinking 等其他块类型，可按需扩展到 metadata。

        usage = response.get("usage", {})
        return cls.assistant(
            content=blocks,
            metadata={
                "model": response.get("model", ""),
                "stop_reason": response.get("stop_reason", ""),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )

    # ==========================================
    #                 展示方法
    # ==========================================
    def __repr__(self) -> str:
        """返回用于调试展示的简短字符串表示。"""
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        tc_info = f", tool_calls={len(self.tool_calls)}" if self.has_tool_calls else ""
        return f"Msg(role={self.role.value!r}, text={preview!r}{tc_info})"


# ==========================================
#                会话消息序列
# ==========================================


class Conversation(BaseModel):
    """组成会话的有序消息集合。

    提供构建、裁剪与序列化工具，用于组织发送给 LLM API 的
    对话历史。

    Attributes:
        messages: 按时间顺序排列的消息列表。
    """

    messages: list[Msg] = Field(default_factory=list)

    # ==========================================
    #                 写入方法
    # ==========================================
    def add(self, msg: Msg) -> None:
        """向会话末尾追加一条消息。"""
        self.messages.append(msg)

    def add_many(self, msgs: Sequence[Msg]) -> None:
        """一次性追加多条消息。"""
        self.messages.extend(msgs)

    # ==========================================
    #                 查询属性
    # ==========================================
    @property
    def system_prompt(self) -> str | None:
        """返回第一条系统消息文本；若不存在则返回 None。"""
        for msg in self.messages:
            if msg.role == Role.SYSTEM:
                return msg.text
        return None

    @property
    def non_system_messages(self) -> list[Msg]:
        """除系统提示外的全部消息（用于 API 的 `messages` 参数）。"""
        return [m for m in self.messages if m.role != Role.SYSTEM]

    @property
    def last(self) -> Msg | None:
        """返回最近一条消息；若为空则返回 None。"""
        return self.messages[-1] if self.messages else None

    @property
    def turn_count(self) -> int:
        """用户消息数量（即可视作会话轮数）。"""
        return sum(
            1 for m in self.messages if m.role == Role.USER and not m.tool_results
        )

    # ==========================================
    #                API 序列化
    # ==========================================
    def to_openai(self) -> list[dict[str, Any]]:
        """将整个会话序列化为 OpenAI API 所需格式。

        Returns:
            扁平化的消息字典列表。
        """
        result: list[dict[str, Any]] = []
        for msg in self.messages:
            converted = msg.to_openai()
            result.extend(converted)
        return result

    def to_anthropic(self) -> dict[str, Any]:
        """将整个会话序列化为 Anthropic API 调用参数。

        Returns:
            含 `system`（字符串）与 `messages`（列表）字段的字典。
        """
        return {
            "system": self.system_prompt or "",
            "messages": [m.to_anthropic() for m in self.non_system_messages],
        }

    # ==========================================
    #                上下文管理
    # ==========================================
    def estimate_tokens(self, chars_per_token: int = 4) -> int:
        """为上下文窗口管理提供粗略 token 数估算。

        这是快速启发式估算，不是精确分词器统计。
        适合用于压缩阈值判断，不适合作为计费依据。

        Args:
            chars_per_token: 每个 token 的平均字符数，默认 4。

        Returns:
            整个会话的估算 token 总数。
        """
        total_chars = sum(len(m.text) for m in self.messages)
        # 计入 tool_use 块中序列化参数带来的额外字符开销。
        for m in self.messages:
            for tc in m.tool_calls:
                total_chars += len(_json_dumps(tc.input))
        return total_chars // chars_per_token

    def slice_recent(self, n: int) -> list[Msg]:
        """返回最近 `n` 条非系统消息。

        可用于构建会话历史的滑动窗口。
        """
        return self.non_system_messages[-n:]

    def clear(self, keep_system: bool = True) -> None:
        """清空消息；可选保留系统提示。"""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == Role.SYSTEM]
        else:
            self.messages.clear()

    # ==========================================
    #                 协议方法
    # ==========================================
    def __len__(self) -> int:
        """返回会话中的消息数量。"""
        return len(self.messages)

    def __iter__(self):
        """返回消息列表迭代器。"""
        return iter(self.messages)


# endregion

# ==========================================
#                 辅助函数
# ==========================================


def _json_dumps(obj: Any) -> str:
    """紧凑 JSON 序列化。"""

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
