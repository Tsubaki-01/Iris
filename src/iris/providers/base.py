"""大模型提供商消息格式接口定义的基类。

定义了不同大模型提供商（如 OpenAI、Anthropic）消息格式之间进行转换的抽象基类。

Example:
    class MyProviderAdapter(MessageAdapter):
        def to_provider(self, msg: Msg) -> Any:
            pass
        def from_provider(self, response: dict[str, Any]) -> Msg:
            pass
"""

# region imports
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..message import ContentBlock, Msg

# endregion


class LLMResponse(BaseModel):
    """提供商中立的 LLM 响应对象，从原始模型 API 载荷解析而来。

    大模型提供商的适配器使用此类作为结构化的中间格式，
    并在后续将模型输出转换为系统统一的 `Msg` 传输对象。

    Attributes:
        content (list[ContentBlock]): 解析后的模型内容块。
        provider (str): 大模型提供商名称（例如 "openai", "anthropic"）。
        id (str): 响应的唯一标识符。
        model (str): 生成此响应的确切模型版本。
        finish_reason (str): 模型停止生成的原因（例如 "stop", "length"）。
        input_tokens (int): 提示词使用的 token 数量。
        output_tokens (int): 生成的 token 数量。
        total_tokens (int): 消耗的 token 总数。
        reasoning (str): 模型的推理过程内容。
        metadata (dict[str, Any]): 特定于提供商的附加元数据。

    Example:
        >>> response = LLMResponse(provider="openai", model="gpt-4o", content=[])
        >>> msg = response.to_msg()
    """

    content: list[ContentBlock] = Field(default_factory=list)
    provider: str = ""
    id: str = ""
    model: str = ""
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_msg(self) -> Msg:
        """将标准化的模型响应转换为助手消息对象。

        提取 token 使用情况和其他相关元数据，将其合并至 metadata 字典中，
        并构建返回统一格式的助手 `Msg` 对象。

        Returns:
            Msg: 包含模型内容与相关元数据的助手消息对象。

        Example:
            >>> response = LLMResponse(provider="openai")
            >>> msg = response.to_msg()
        """

        metadata: dict[str, Any] = {
            "provider": self.provider,
            "id": self.id,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.reasoning:
            metadata["reasoning"] = self.reasoning
        metadata.update(self.metadata)
        return Msg.assistant(content=self.content, metadata=metadata)


class MessageAdapter(BaseModel):
    """用于特定大模型提供商消息格式适配器的基类。

    该类定义了标准化消息对象 (Msg) 与特定大模型 API 所需的原始格式之间转换的规范协议。

    Example:
        >>> adapter = OpenAIMessageAdapter()
        >>> api_payload = adapter.to_provider(msg)
        >>> response = adapter.from_provider(api_response)
    """

    @abstractmethod
    def to_provider(self, msg: Msg) -> Any:
        """将 Iris 标准消息转换为提供商 API 所需的特定格式。

        Args:
            msg (Msg): Iris 标准化的消息对象。

        Returns:
            Any: 大模型提供商 API 可接受的数据结构（通常为字典或列表）。

        Raises:
            NotImplementedError: 子类必须实现此方法。

        Example:
            >>> payload = adapter.to_provider(Msg.user("Hello"))
        """
        raise NotImplementedError

    @abstractmethod
    def from_provider(self, response: dict[str, Any]) -> LLMResponse:
        """将提供商 API 的响应结果解析为 Iris 标准消息对象。

        Args:
            response (dict[str, Any]): 原始的大模型提供商 API 响应字典。

        Returns:
            LLMResponse: 统一封装的 Iris LLM 响应对象。

        Raises:
            NotImplementedError: 子类必须实现此方法。

        Example:
            >>> response = adapter.from_provider({"choices": []})
        """
        raise NotImplementedError
