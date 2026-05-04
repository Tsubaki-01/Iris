"""LLM 调用请求与响应模型。

本模块定义 provider-neutral 的调用边界：`LLMRequest` 描述一次模型调用，
`LLMResponse` 描述厂商响应标准化后的结果。Provider 层只能导入这些模型，
不能在自身模块里重新定义或重导出同名模型。

Example:
    >>> request = LLMRequest(model="gpt-4o", messages=[Msg.user("你好")])
    >>> response = LLMResponse(provider="openai", content=[TextBlock(text="你好")])
    >>> response.to_msg().text
    '你好'
"""

# region imports
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field

from .message import ContentBlock, Msg, Role

if TYPE_CHECKING:
    from .message import Conversation
# endregion

__all__ = ["LLMRequest", "LLMResponse"]


class LLMRequest(BaseModel):
    """一次 LLM 调用请求。

    `LLMRequest` 是调用级模型，不是 `Conversation` 的别名。它把会话历史、
    模型参数、工具定义和少量厂商选项放在同一个 provider-neutral 对象里，
    让 adapter 可以只依赖一个稳定输入。

    Attributes:
        model (str): 模型名称。
        messages (list[Msg]): 本次调用发送给模型的消息历史。
        temperature (float | None): 采样温度。
        top_p (float | None): nucleus sampling 参数。
        max_tokens (int | None): 最大输出 token 数。
        tools (list[dict[str, Any]]): 可用工具定义。
        tool_choice (str | dict[str, Any] | None): 工具选择策略。
        response_format (dict[str, Any] | None): 结构化输出配置。
        stream (bool): 是否请求流式响应。
        timeout (float | None): 单次请求超时时间，单位秒。
        provider_options (dict[str, Any]): 少量 provider 专属选项。
        metadata (dict[str, Any]): 请求级元数据。

    Example:
        >>> conversation = Conversation(messages=[Msg.user("你好")])
        >>> request = LLMRequest.from_conversation(conversation, "gpt-4o")
        >>> request.model
        'gpt-4o'
    """

    model: str
    messages: list[Msg] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    timeout: float | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_conversation(
        cls,
        conversation: Conversation,
        model: str,
        **options: Any,
    ) -> Self:
        """从会话构建一次 LLM 请求。

        Args:
            conversation (Conversation): 提供消息历史的会话对象。
            model (str): 本次调用使用的模型名称。
            **options (Any): 传递给 `LLMRequest` 的请求级参数。

        Returns:
            Self: 包含会话消息快照的新请求对象。

        Example:
            >>> conversation = Conversation(messages=[Msg.user("你好")])
            >>> LLMRequest.from_conversation(conversation, "gpt-4o").messages
            [Msg(role='user', text='你好')]
        """
        return cls(model=model, messages=list(conversation.messages), **options)

    def system_prompt(self) -> str | None:
        """返回第一条系统消息文本。

        Returns:
            str | None: 第一条 system 消息文本；没有 system 消息时返回 None。

        Example:
            >>> LLMRequest(model="gpt-4o", messages=[Msg.system("规则")]).system_prompt()
            '规则'
        """
        for msg in self.messages:
            if msg.role == Role.SYSTEM:
                return msg.text
        return None

    def non_system_messages(self) -> list[Msg]:
        """返回排除系统消息后的消息列表。

        Anthropic 等 API 会把 system prompt 放到顶层字段，因此 adapter 需要
        一个明确入口获取非 system 消息。

        Returns:
            list[Msg]: 保持原顺序的非 system 消息列表。

        Example:
            >>> request = LLMRequest(
            ...     model="gpt-4o",
            ...     messages=[Msg.system("规则"), Msg.user("你好")],
            ... )
            >>> [msg.role for msg in request.non_system_messages()]
            [<Role.USER: 'user'>]
        """
        return [msg for msg in self.messages if msg.role != Role.SYSTEM]


class LLMResponse(BaseModel):
    """Provider-neutral LLM 响应。

    Adapter 将厂商 raw response 解析成此模型后，上层只需要处理统一字段。
    厂商特有字段应放入 `metadata`，避免污染 `Msg` 或业务主流程。

    Attributes:
        provider (str): 厂商名称，例如 `"openai"` 或 `"anthropic"`。
        id (str): 厂商响应 ID。
        model (str): 实际返回响应的模型名称。
        content (list[ContentBlock]): 文本、工具调用等标准内容块。
        finish_reason (str): 模型停止生成的原因。
        input_tokens (int): 输入 token 数。
        output_tokens (int): 输出 token 数。
        total_tokens (int): token 总数。
        reasoning (str): 推理摘要或推理文本。
        metadata (dict[str, Any]): 厂商特有或追踪相关元数据。

    Example:
        >>> response = LLMResponse(provider="openai", content=[TextBlock(text="你好")])
        >>> response.to_msg().role
        <Role.ASSISTANT: 'assistant'>
    """

    provider: str
    id: str = ""
    model: str = ""
    content: list[ContentBlock] = Field(default_factory=list)
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_msg(self) -> Msg:
        """转换为 Iris 内部助手消息。

        Provider 字段统一进入 `Msg.metadata`，这样 `Msg` 可以保持稳定，
        不会随着 OpenAI、Anthropic 等厂商字段变化而扩张。

        Returns:
            Msg: 包含标准内容块和响应元数据的 assistant 消息。

        Example:
            >>> LLMResponse(provider="openai", content=[TextBlock(text="你好")]).to_msg().text
            '你好'
        """
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "id": self.id,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
        }
        if self.reasoning:
            metadata["reasoning"] = self.reasoning
        metadata.update(self.metadata)
        return Msg.assistant(content=self.content, metadata=metadata)
