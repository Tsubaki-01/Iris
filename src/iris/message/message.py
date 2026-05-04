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
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

# endregion

if TYPE_CHECKING:
    from .llm import LLMRequest

# region definitions
# ==========================================
#                 枚举定义
# ==========================================


class Role(StrEnum):
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
# region Msg
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
        parts = [block.text for block in self.content if isinstance(block, TextBlock)]
        return "\n".join(parts)

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        """提取助手消息中的所有工具调用块。"""
        if isinstance(self.content, str):
            return []
        return [block for block in self.content if isinstance(block, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        """提取所有工具结果块。一般来说一条Msg中只包含一个tool_result。"""
        if isinstance(self.content, str):
            return []
        return [block for block in self.content if isinstance(block, ToolResultBlock)]

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Msg:
        """从字典构建一个Msg对象，并为其内容块指定类型。"""
        content = data.get("content", "")
        if isinstance(content, list):
            content = [_content_block_from_dict(block) for block in content]

        return cls(
            role=data["role"],
            content=content,
            sender=data.get("sender", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )

    # ==========================================
    #                 展示方法
    # ==========================================
    def __repr__(self) -> str:
        """返回用于调试展示的简短字符串表示。"""
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        tc_info = f", tool_calls={len(self.tool_calls)}" if self.has_tool_calls else ""
        return f"Msg(role={self.role.value!r}, text={preview!r}{tc_info})"


# endregion

# region Conversation
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
        return [msg for msg in self.messages if msg.role != Role.SYSTEM]

    @property
    def last(self) -> Msg | None:
        """返回最近一条消息；若为空则返回 None。"""
        return self.messages[-1] if self.messages else None

    @property
    def turn_count(self) -> int:
        """用户消息数量（即可视作会话轮数）。"""
        return sum(
            1 for msg in self.messages if msg.role == Role.USER and not msg.tool_results
        )

    # ==========================================
    #                API 序列化
    # ==========================================
    def to_llm_request(self, model: str, **options: Any) -> LLMRequest:
        """将当前会话构建为一次 LLM 调用请求。

        `Conversation` 只负责内部会话历史，不直接生成 OpenAI、Anthropic
        等厂商 payload。具体 provider 格式转换由 providers adapter 负责。

        Args:
            model: 本次调用使用的模型名称。
            **options: 传递给 `LLMRequest` 的请求级参数。

        Returns:
            一次 provider-neutral 的 LLM 调用请求。
        """
        from .llm import LLMRequest

        return LLMRequest.from_conversation(self, model, **options)

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
        total_chars = sum(len(msg.text) for msg in self.messages)
        # 计入 tool_use 块中序列化参数带来的额外字符开销。
        for msg in self.messages:
            for tool_call in msg.tool_calls:
                total_chars += len(_json_dumps(tool_call.input))
        return total_chars // chars_per_token

    def slice_recent(self, n: int) -> list[Msg]:
        """返回最近 `n` 条非系统消息。

        可用于构建会话历史的滑动窗口。
        """
        return self.non_system_messages[-n:]

    def clear(self, keep_system: bool = True) -> None:
        """清空消息；可选保留系统提示。"""
        if keep_system:
            self.messages = [msg for msg in self.messages if msg.role == Role.SYSTEM]
        else:
            self.messages.clear()

    # ==========================================
    #                 协议方法
    # ==========================================
    def __len__(self) -> int:
        """返回会话中的消息数量。"""
        return len(self.messages)


# endregion
# endregion

# ==========================================
#                 辅助函数
# ==========================================


def _json_dumps(obj: Any) -> str:
    """紧凑 JSON 序列化。"""

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _content_block_from_dict(data: dict[str, Any]) -> ContentBlock:
    block_type = data.get("type")
    if block_type == "text":
        return TextBlock.model_validate(data)
    if block_type == "tool_use":
        return ToolUseBlock.model_validate(data)
    if block_type == "tool_result":
        return ToolResultBlock.model_validate(data)
    raise ValueError(f"Unsupported content block type: {block_type!r}")
