"""Provider HTTP 调用客户端。

`ProviderClient` 组合 `ProviderAdapter` 与 `httpx.AsyncClient`，负责真实
HTTP 调用、鉴权 header、endpoint 选择和错误映射。格式转换仍由 adapter
完成，避免网络层与 provider payload 解析混在一起。

Example:
    >>> from iris.providers import OpenAIMessageAdapter, ProviderClient
    >>> client = ProviderClient(adapter=OpenAIMessageAdapter(), api_key="test")
    >>> client.adapter.provider
    'openai'
"""

# region imports
from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import (
    IrisAPIConnectionError,
    IrisAuthenticationError,
    IrisProviderError,
    IrisRateLimitExceededError,
)
from ..message.llm import LLMRequest, LLMResponse
from .adapter import ProviderAdapter

# endregion

# ==========================================
#                 Constants
# ==========================================
# region constants
OPENAI_BASE_URL = "https://api.openai.com"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
# endregion


class ProviderClient(BaseModel):
    """真实 provider HTTP 调用层。

    Client 只负责传输层问题：base URL、headers、HTTP status、连接错误。
    请求 payload 与响应解析通过注入的 `ProviderAdapter` 完成。

    Attributes:
        adapter (ProviderAdapter): 当前 provider 的格式适配器。
        api_key (str): Provider API key。
        base_url (str | None): 自定义 provider base URL；为空时使用内置默认值。
        timeout (float | None): 默认请求超时时间，单位秒。
        http_client (httpx.AsyncClient | None): 可注入的 HTTP client，便于测试。
        headers (dict[str, str]): 追加或覆盖的 HTTP headers。

    Example:
        >>> from iris.providers import OpenAIMessageAdapter
        >>> client = ProviderClient(adapter=OpenAIMessageAdapter(), api_key="test")
        >>> client._endpoint_for(LLMRequest(model="gpt-4o"))
        '/v1/chat/completions'
    """

    adapter: ProviderAdapter
    api_key: str
    base_url: str | None = None
    timeout: float | None = None
    http_client: httpx.AsyncClient | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """发送非流式 LLM 请求并返回标准响应。

        Args:
            request (LLMRequest): 一次模型调用请求。

        Returns:
            LLMResponse: Adapter 解析后的 provider-neutral 响应。

        Raises:
            IrisProviderError: 传入 `stream=True` 或 provider 不受支持时抛出。
            IrisAPIConnectionError: HTTP 连接或传输失败时抛出。
            IrisAuthenticationError: Provider 返回 401 或 403 时抛出。
            IrisRateLimitExceededError: Provider 返回 429 时抛出。

        Example:
            >>> request = LLMRequest(model="gpt-4o")
            >>> request.stream
            False
        """
        if request.stream:
            raise IrisProviderError("complete() 不支持 stream=True，请使用后续 stream() 接口")
        payload = self.adapter.to_provider_request(request)
        raw_response = await self._send(request, payload)
        return self.adapter.from_provider_response(raw_response)

    async def close(self) -> None:
        """关闭内部 HTTP client。

        Returns:
            None: 此方法只释放 client 资源。

        Example:
            >>> # await client.close()
        """
        if self.http_client is not None:
            await self.http_client.aclose()

    async def _send(self, request: LLMRequest, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 HTTP 请求并返回 JSON 字典。"""
        endpoint = self._endpoint_for(request)
        url = f"{self._base_url().rstrip('/')}{endpoint}"
        client = self._client_for_request(request)
        try:
            response = await client.post(url, json=payload, headers=self._build_headers())
        except httpx.HTTPError as exc:
            raise IrisAPIConnectionError(
                "连接 provider API 失败",
                provider=self.adapter.provider,
            ) from exc
        if response.status_code >= 400:
            raise self._map_http_error(response.status_code, self._response_body(response))
        body = response.json()
        return body if isinstance(body, dict) else {"data": body}

    def _client_for_request(self, request: LLMRequest) -> httpx.AsyncClient:
        """返回本次请求使用的 HTTP client。"""
        if self.http_client is not None:
            return self.http_client
        timeout = request.timeout if request.timeout is not None else self.timeout
        self.http_client = httpx.AsyncClient(base_url=self._base_url(), timeout=timeout)
        return self.http_client

    def _endpoint_for(self, request: LLMRequest) -> str:
        """根据 provider 与 API 风格选择 endpoint。"""
        if self.adapter.provider == "openai":
            api_style = request.provider_options.get("api_style", self.adapter.default_api_style)
            return "/v1/responses" if api_style == "responses" else "/v1/chat/completions"
        if self.adapter.provider == "anthropic":
            return "/v1/messages"
        raise IrisProviderError("不支持的 provider", provider=self.adapter.provider)

    def _build_headers(self) -> dict[str, str]:
        """构建 provider 鉴权与内容类型 headers。"""
        headers = {"Content-Type": "application/json", **self.headers}
        if self.adapter.provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.adapter.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        return headers

    def _map_http_error(self, status_code: int, body: Any) -> IrisProviderError:
        """将 HTTP status 映射为 Iris provider 异常。"""
        message = self._error_message(body)
        if status_code in {401, 403}:
            return IrisAuthenticationError(
                message,
                status_code=status_code,
                provider=self.adapter.provider,
            )
        if status_code == 429:
            return IrisRateLimitExceededError(
                message,
                status_code=status_code,
                provider=self.adapter.provider,
            )
        return IrisProviderError(message, status_code=status_code, provider=self.adapter.provider)

    def _response_body(self, response: httpx.Response) -> Any:
        """提取 HTTP 错误响应体。"""
        try:
            return response.json()
        except ValueError:
            return response.text

    def _error_message(self, body: Any) -> str:
        """从 provider 错误响应中提取可读错误信息。"""
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if body.get("message"):
                return str(body["message"])
        if isinstance(body, str) and body:
            return body
        return "provider API 调用失败"

    def _base_url(self) -> str:
        """返回当前 provider 的 base URL。"""
        if self.base_url:
            return self.base_url
        if self.adapter.provider == "anthropic":
            return ANTHROPIC_BASE_URL
        return OPENAI_BASE_URL
