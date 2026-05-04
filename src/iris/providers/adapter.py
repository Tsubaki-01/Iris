"""Provider 格式适配器基类。

本模块只定义 provider adapter 的抽象边界。LLM 请求与响应模型只在
`iris.message.llm` 中定义，并通过 `iris.message` 导出，避免 provider 层
出现重复模型或兼容别名。

Example:
    >>> class EchoAdapter(ProviderAdapter):
    ...     provider = "echo"
    ...     def to_provider_request(self, request):
    ...         return {"model": request.model}
    ...     def from_provider_response(self, response):
    ...         raise NotImplementedError
"""

# region imports
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from ..message.llm import LLMRequest, LLMResponse

# endregion


class ProviderAdapter(BaseModel):
    """Provider 格式适配器基类。

    Adapter 只做纯数据格式转换，不读取 API key，不创建 HTTP client，
    不执行重试或错误映射。这些职责属于 `ProviderClient`。

    Attributes:
        provider (str): Provider 名称，例如 `"openai"` 或 `"anthropic"`。
        default_api_style (str | None): Provider 默认 API 风格。

    Example:
        >>> adapter = ProviderAdapter(provider="base")
        Traceback (most recent call last):
        ...
        TypeError: Can't instantiate abstract class
    """

    provider: str
    default_api_style: str | None = None

    @abstractmethod
    def to_provider_request(self, request: LLMRequest) -> dict[str, Any]:
        """将 Iris 请求转换为 provider payload。

        Args:
            request (LLMRequest): Provider-neutral 的一次 LLM 调用请求。

        Returns:
            dict[str, Any]: 可直接发送给厂商 API 的请求 payload。

        Raises:
            NotImplementedError: 子类未实现转换逻辑时抛出。
        """
        raise NotImplementedError

    @abstractmethod
    def from_provider_response(self, response: Mapping[str, Any]) -> LLMResponse:
        """将 provider raw response 转换为标准响应。

        Args:
            response (Mapping[str, Any]): 厂商 API 返回的原始 JSON 对象。

        Returns:
            LLMResponse: Provider-neutral 响应模型。

        Raises:
            NotImplementedError: 子类未实现解析逻辑时抛出。
        """
        raise NotImplementedError


__all__ = ["ProviderAdapter"]
