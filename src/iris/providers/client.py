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

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast
from uuid import uuid4

import litellm
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from ..message.llm import LLMRequest, LLMResponse
from ..message.streaming import ModelStreamEvent, ModelStreamScope
from ._streaming import _iter_litellm_events, failed_before_start
from .openai import OpenAIChatMapper

# endregion


class ProviderClient(BaseModel):
    """Provider Chat Completion 调用层。

    Client 只负责将 Iris 的 provider-neutral 请求转换成 LiteLLM chat kwargs，
    并把响应和异常映射回 Iris 边界。

    Attributes:
        provider (str): Provider 名称，例如 `"openai"` 或 `"anthropic"`。
        litellm_provider (str | None): 实际传给 LiteLLM 的 provider 名称。
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
    litellm_provider: str | None = None
    api_key: str
    base_url: str | None = None
    timeout: float | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """使用实际 Chat 消息和工具定义估算完整输入，不发起生成请求。

        Args:
            request: 已应用模型选项和工具 schema 的最终请求。

        Returns:
            输入 token 估算值，包含 response_format 的序列化文本。

        Raises:
            IrisProviderError: 请求风格无效或底层计量失败。
        """
        self._validate_api_style(request)
        model = request.model.removeprefix(f"{self.litellm_provider or self.provider}/")
        try:
            count = litellm.token_counter(
                model=model,
                messages=OpenAIChatMapper().format_messages(request.messages),
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            if request.response_format is not None:
                count += litellm.token_counter(
                    model=model,
                    text=json.dumps(request.response_format, ensure_ascii=False),
                )
            return count
        except Exception as exc:
            raise self._map_litellm_error(exc) from exc

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
                "complete() 不支持 stream=True，请使用 stream() 接口",
                provider=self.provider,
            )
        self._validate_api_style(request)
        try:
            response = await litellm.acompletion(**self._to_litellm_kwargs(request))
        except Exception as exc:
            raise self._map_litellm_error(exc) from exc
        return self._from_litellm_response(response)

    async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """发送流式Chat Completion请求并产出标准事件。

        Args:
            request: `stream=True`的provider-neutral模型请求。

        Yields:
            不包含LiteLLM raw对象的连续模型流式事件。

        Raises:
            IrisProviderError: 请求未启用stream或使用非Chat API风格。
            asyncio.CancelledError: 本地consumer task被取消。
        """
        if not request.stream:
            raise IrisProviderError(
                "stream() 不支持 stream=False",
                provider=self.provider,
            )
        self._validate_api_style(request)
        scope = ModelStreamScope(
            model_stream_id=f"model-stream-{uuid4().hex}",
            provider=self.provider,
            model=request.model,
            attempt=1,
        )
        kwargs = self._to_litellm_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        try:
            raw_stream = await litellm.acompletion(**kwargs)
        except Exception as exc:
            yield failed_before_start(
                scope=scope,
                error=self._map_litellm_error(exc),
                as_mapping=self._as_mapping,
            )
            return

        async for event in _iter_litellm_events(
            cast(AsyncIterator[Any], raw_stream),
            scope=scope,
            as_mapping=self._as_mapping,
            error_mapper=self._map_litellm_error,
        ):
            yield event

    def _validate_api_style(self, request: LLMRequest) -> None:
        """在原始 provider 选项边界拒绝不支持的非 Chat 请求。"""
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
            "messages": OpenAIChatMapper().format_messages(
                request.messages,
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

        for name in ("reasoning_effort", "num_retries"):
            if name in request.provider_options:
                kwargs[name] = request.provider_options[name]
        return kwargs

    def _litellm_model(self, model: str) -> str:
        """返回 LiteLLM 需要的 provider/model 模型名。"""
        provider = self.litellm_provider or self.provider
        if model.split("/", 1)[0] == provider:
            return model
        return f"{provider}/{model}"

    def _from_litellm_response(self, response: Any) -> LLMResponse:
        """将 LiteLLM Chat Completion 响应转换为 Iris 标准响应。"""
        data = self._as_mapping(response)
        choices = self._get(data, "choices", []) or []
        choice = choices[0] if choices else {}
        message = self._get(choice, "message", {}) or {}
        usage = self._get(data, "usage", {}) or {}
        raw_object = self._get(data, "object", "")
        reasoning = self._get(message, "reasoning_content", self._get(message, "reasoning", ""))
        return LLMResponse(
            provider=self.provider,
            id=str(self._get(data, "id", "") or ""),
            model=str(self._get(data, "model", "") or ""),
            content=OpenAIChatMapper().content_blocks_from_chat_message(self._as_mapping(message)),
            finish_reason=str(self._get(choice, "finish_reason", "") or ""),
            input_tokens=int(self._get(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(self._get(usage, "completion_tokens", 0) or 0),
            total_tokens=int(self._get(usage, "total_tokens", 0) or 0),
            reasoning=reasoning if isinstance(reasoning, str) else "",
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
        return {}

    def _get(self, value: Any, key: str, default: Any = None) -> Any:
        """兼容 Mapping 与对象属性读取。"""
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)
