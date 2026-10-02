"""Provider 双协议调用客户端。

`ProviderClient` 是 Iris provider-neutral 请求与 LiteLLM Responses / Chat Completions
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

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import aclosing
from typing import Any, Literal, cast
from uuid import uuid4

import litellm
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from ..message.llm import LLMRequest, LLMResponse
from ..message.streaming import ModelStreamEvent, ModelStreamScope
from ._chat_completions_streaming import _iter_chat_completions_events
from ._responses_streaming import _iter_responses_events
from ._stream_utils import failed_before_start
from .chat_completions import ChatCompletionsAdapter
from .responses import ResponsesAdapter

# endregion


class ProviderClient(BaseModel):
    """Provider 双协议调用层。

    Client 在构造时选定固定 adapter，将 Iris 请求转换成对应协议参数，
    并把响应和异常映射回 Iris 边界。

    Attributes:
        provider (str): 用于配置和凭据查找的逻辑 provider 名称，例如 `"openai"` 或 `"anthropic"`。
        litellm_provider (str | None): LiteLLM 传输 provider；与 api_style 协议选择分开。
        api_style (str): 构造时选定的 responses 或 chat_completions。
        api_key (str): Provider API key。
        base_url (str | None): 自定义 provider base URL。
        timeout (float | None): 默认请求超时时间，单位秒。
        headers (dict[str, str]): 透传给 provider 的额外 headers。

    Example:
        >>> client = ProviderClient(provider="openai", api_key="test")
        >>> client.provider
        'openai'
    """

    provider: str
    litellm_provider: str | None = None
    api_style: Literal["responses", "chat_completions"] = "responses"
    api_key: str
    base_url: str | None = None
    timeout: float | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _adapter: ResponsesAdapter | ChatCompletionsAdapter = PrivateAttr()
    _stream_iterator: Callable[..., AsyncGenerator[ModelStreamEvent, None]] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        """构造时选定协议，后续请求不重新选择或回退。"""
        if self.api_style == "responses":
            self._adapter = ResponsesAdapter()
            self._stream_iterator = _iter_responses_events
        else:
            self._adapter = ChatCompletionsAdapter()
            self._stream_iterator = _iter_chat_completions_events

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """从所选协议的实际输入投影本地计量输入，不发起生成请求。

        Args:
            request: 已应用模型选项和逻辑工具定义的最终请求。

        Returns:
            输入 token 估算值，包含 response_format 的序列化文本。

        Raises:
            IrisProviderError: 请求风格无效或底层计量失败。
        """
        model = request.model.removeprefix(f"{self.litellm_provider or self.provider}/")
        try:
            count = litellm.token_counter(
                model=model,
                **self._adapter.token_count_projection(
                    request, transport=self.litellm_provider or self.provider
                ),
            )
            if request.response_format is not None:
                count += litellm.token_counter(
                    model=model,
                    text=json.dumps(
                        self._adapter.format_for_count(request.response_format), ensure_ascii=False
                    ),
                )
            return count
        except Exception as exc:
            raise self._map_provider_error(exc) from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """发送选定协议的非流式请求并返回标准响应。

        Args:
            request (LLMRequest): 一次模型调用请求。

        Returns:
            LLMResponse: 解析后的 provider-neutral 响应。

        Raises:
            IrisProviderError: 协议或传输不支持，或响应未完成时抛出。
            IrisAPIConnectionError: 连接或超时时抛出。
            IrisAuthenticationError: Provider 返回认证错误时抛出。
            IrisRateLimitExceededError: Provider 返回限流错误时抛出。

        Example:
            >>> request = LLMRequest(model="gpt-4o")
            >>> request.stream
            False
        """
        if request.stream:
            raise IrisProviderError(
                "complete() 不支持 stream=True，请使用 stream() 接口",
                provider=self.provider,
            )
        try:
            response = await self._adapter.invoke(self._to_adapter_kwargs(request))
            return self._parse_response(response)
        except Exception as exc:
            raise self._map_provider_error(exc) from exc

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ModelStreamEvent, None]:
        """发送选定协议的流式请求并产出标准事件。

        Args:
            request: `stream=True`的provider-neutral模型请求。

        Yields:
            不包含 LiteLLM 原始对象的连续模型流式事件。

        Raises:
            IrisProviderError: 请求未启用 stream。
            asyncio.CancelledError: 本地consumer task被取消。
        """
        if not request.stream:
            raise IrisProviderError(
                "stream() 不支持 stream=False",
                provider=self.provider,
            )
        scope = ModelStreamScope(
            model_stream_id=f"model-stream-{uuid4().hex}",
            provider=self.provider,
            model=request.model,
            attempt=1,
        )
        try:
            kwargs = self._to_adapter_kwargs(request)
            kwargs["stream"] = True
            raw_stream = await self._adapter.invoke(kwargs)
        except Exception as exc:
            yield failed_before_start(
                scope=scope,
                error=self._map_provider_error(exc),
            )
            return

        async with aclosing(
            self._stream_iterator(
                cast(AsyncIterator[Any], raw_stream),
                scope=scope,
                as_mapping=self._as_mapping,
                error_mapper=self._map_provider_error,
            )
        ) as events:
            async for event in events:
                yield event

    def _to_adapter_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        """组合公共调用选项和固定 adapter 生成的协议参数。"""
        kwargs: dict[str, Any] = {
            "model": self._routed_model(request.model),
            "api_key": self.api_key,
            **self._adapter.encode_request(
                request, transport=self.litellm_provider or self.provider
            ),
        }
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.headers:
            kwargs["extra_headers"] = self.headers
        for name in ("temperature", "top_p"):
            value = getattr(request, name)
            if value is not None:
                kwargs[name] = value
        timeout = request.timeout if request.timeout is not None else self.timeout
        if timeout is not None:
            kwargs["timeout"] = timeout
        if "num_retries" in request.provider_options:
            kwargs["num_retries"] = request.provider_options["num_retries"]
        return kwargs

    def _routed_model(self, model: str) -> str:
        """返回包含传输 provider 前缀的模型路由。"""
        provider = self.litellm_provider or self.provider
        if model.split("/", 1)[0] == provider:
            return model
        return f"{provider}/{model}"

    def _parse_response(self, response: Any) -> LLMResponse:
        """将所选协议的完整响应转换为 Iris 标准响应。"""
        return self._adapter.parse_response(self._as_mapping(response), provider=self.provider)

    def _map_provider_error(self, exc: Exception) -> IrisProviderError:
        """将 LiteLLM 调用异常映射为 Iris provider 异常。"""
        if isinstance(exc, IrisProviderError):
            return exc
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
        """从 LiteLLM 调用异常或其 response 中提取 HTTP status。"""
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    def _as_mapping(self, value: Any) -> Mapping[str, Any]:
        """在响应边界将 dict 或 LiteLLM/Pydantic 对象转换为 Mapping。"""
        if isinstance(value, Mapping):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, Mapping) else {}
        return {}
