"""Provider LiteLLM 调用客户端。

`ProviderClient` 是 Iris provider-neutral 请求与 LiteLLM Chat Completion
之间的边界。它保留 Iris 自己的 `LLMRequest`、`LLMResponse` 和异常类型，
不把 LiteLLM 对象向上传递。

Example:
    >>> from iris.providers import ProviderClient
    >>> client = ProviderClient(provider="openai", api_key="test")
    >>> client.provider
    'openai'
"""

# region imports
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import litellm
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from ..message.llm import LLMRequest, LLMResponse
from .adapter import ProviderAdapter
from .anthropic import AnthropicMessageAdapter
from .openai import OpenAIMessageAdapter

# endregion


class ProviderClient(BaseModel):
    """Provider Chat Completion 调用层。

    Client 只负责将 Iris 的 provider-neutral 请求转换成 LiteLLM chat kwargs，
    并把响应和异常映射回 Iris 边界。`adapter` 构造参数仅作为旧 factory 的
    兼容输入，不再是 active model field。

    Attributes:
        provider (str): Provider 名称，例如 `"openai"` 或 `"anthropic"`。
        api_key (str): Provider API key。
        base_url (str | None): 自定义 provider base URL。
        timeout (float | None): 默认请求超时时间，单位秒。
        headers (dict[str, str]): 透传给 LiteLLM 的额外 headers。

    Example:
        >>> client = ProviderClient(provider="openai", api_key="test")
        >>> client.provider
        'openai'
    """

    provider: str
    api_key: str
    base_url: str | None = None
    timeout: float | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _derive_provider_from_legacy_adapter(cls, data: Any) -> Any:
        """从旧 `adapter=` 构造输入派生 provider。"""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        adapter = normalized.get("adapter")
        if "provider" not in normalized and isinstance(adapter, ProviderAdapter):
            normalized["provider"] = adapter.provider
        return normalized

    @property
    def adapter(self) -> ProviderAdapter:
        """返回当前 provider 的兼容格式适配器。"""
        if self.provider == "openai":
            return OpenAIMessageAdapter()
        if self.provider == "anthropic":
            return AnthropicMessageAdapter()
        raise IrisProviderError("不支持的 provider", provider=self.provider)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """发送非流式 Chat Completion 请求并返回标准响应。

        Args:
            request (LLMRequest): 一次模型调用请求。

        Returns:
            LLMResponse: 解析后的 provider-neutral 响应。

        Raises:
            IrisProviderError: 传入 `stream=True` 或非 chat API 风格时抛出。
            IrisAPIConnectionError: LiteLLM 连接或超时时抛出。
            IrisAuthenticationError: Provider 返回认证错误时抛出。
            IrisRateLimitExceededError: Provider 返回限流错误时抛出。

        Example:
            >>> request = LLMRequest(model="gpt-4o")
            >>> request.stream
            False
        """
        if request.stream:
            raise IrisProviderError(
                "complete() 不支持 stream=True，请使用后续 stream() 接口",
                provider=self.provider,
            )
        self._validate_api_style(request)
        try:
            response = await litellm.acompletion(**self._to_litellm_kwargs(request))
        except Exception as exc:
            raise self._map_litellm_error(exc) from exc
        return self._from_litellm_response(response)

    async def close(self) -> None:
        """释放 client 资源。

        LiteLLM 路径不持有 Iris 自己创建的 HTTP client，因此这里保持为
        兼容 no-op。

        Returns:
            None: 无需释放资源。
        """
        return None

    def _validate_api_style(self, request: LLMRequest) -> None:
        """拒绝本阶段不支持的非 Chat API 风格。"""
        api_style = request.provider_options.get("api_style", "chat")
        if api_style != "chat":
            raise IrisProviderError(
                f"不支持的 provider API 风格: {api_style}",
                provider=self.provider,
                api_style=api_style,
            )

    def _to_litellm_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        """将 Iris 请求转换为 LiteLLM `acompletion` kwargs。"""
        kwargs: dict[str, Any] = {
            "model": self._litellm_model(request.model),
            "messages": OpenAIMessageAdapter().format_messages(
                request.messages,
                api_style="chat",
            ),
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.headers:
            kwargs["extra_headers"] = self.headers

        for name in (
            "temperature",
            "top_p",
            "max_tokens",
            "tool_choice",
            "response_format",
        ):
            value = getattr(request, name)
            if value is not None:
                kwargs[name] = value
        if request.tools:
            kwargs["tools"] = request.tools

        timeout = request.timeout if request.timeout is not None else self.timeout
        if timeout is not None:
            kwargs["timeout"] = timeout

        if "reasoning_effort" in request.provider_options:
            kwargs["reasoning_effort"] = request.provider_options["reasoning_effort"]
        return kwargs

    def _litellm_model(self, model: str) -> str:
        """返回 LiteLLM 需要的 provider/model 模型名。"""
        if "/" in model:
            return model
        return f"{self.provider}/{model}"

    def _from_litellm_response(self, response: Any) -> LLMResponse:
        """将 LiteLLM Chat Completion 响应转换为 Iris 标准响应。"""
        data = self._as_mapping(response)
        choices = self._get(data, "choices", []) or []
        choice = choices[0] if choices else {}
        message = self._get(choice, "message", {}) or {}
        usage = self._get(data, "usage", {}) or {}
        raw_object = self._get(data, "object", "")
        return LLMResponse(
            provider=self.provider,
            id=str(self._get(data, "id", "") or ""),
            model=str(self._get(data, "model", "") or ""),
            content=OpenAIMessageAdapter()._content_blocks_from_chat_message(
                self._as_mapping(message)
            ),
            finish_reason=str(self._get(choice, "finish_reason", "") or ""),
            input_tokens=int(self._get(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(self._get(usage, "completion_tokens", 0) or 0),
            total_tokens=int(self._get(usage, "total_tokens", 0) or 0),
            metadata={"raw_object": raw_object} if raw_object else {},
        )

    def _map_litellm_error(self, exc: Exception) -> IrisProviderError:
        """将 LiteLLM/OpenAI 风格异常映射为 Iris provider 异常。"""
        status_code = self._status_code_from_exception(exc)
        message = str(exc) or "provider API 调用失败"
        error_name = exc.__class__.__name__
        if status_code in {401, 403} or error_name in {
            "AuthenticationError",
            "PermissionDeniedError",
        }:
            return IrisAuthenticationError(
                message,
                status_code=status_code,
                provider=self.provider,
            )
        if status_code == 429 or error_name in {
            "RateLimitError",
            "RouterRateLimitError",
        }:
            return IrisRateLimitExceededError(
                message,
                status_code=status_code,
                provider=self.provider,
            )
        if status_code == 408 or error_name in {
            "APIConnectionError",
            "APITimeoutError",
        }:
            return IrisAPIConnectionError(
                message,
                status_code=status_code,
                provider=self.provider,
            )
        return IrisProviderError(
            message,
            status_code=status_code,
            provider=self.provider,
        )

    def _status_code_from_exception(self, exc: Exception) -> int | None:
        """从 LiteLLM 异常或其 response 中提取 HTTP status。"""
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    def _as_mapping(self, value: Any) -> Mapping[str, Any]:
        """将 dict、Pydantic/LiteLLM 对象转换为只读 Mapping 形状。"""
        if isinstance(value, Mapping):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, Mapping) else {}
        if hasattr(value, "dict"):
            dumped = value.dict()
            return dumped if isinstance(dumped, Mapping) else {}
        return {}

    def _get(self, value: Any, key: str, default: Any = None) -> Any:
        """兼容 Mapping 与对象属性读取。"""
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)
