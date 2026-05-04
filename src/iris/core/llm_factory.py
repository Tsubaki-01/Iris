"""LLM provider client 工厂。

本模块负责解析高层模型配置字符串，并按 provider 创建对应的
`ProviderClient`。Adapter 仍只做格式转换，`ProviderClient` 仍只负责
HTTP 传输，本模块只承担高层装配职责。

Example:
    >>> route = parse_model_route("openai/gpt-4o")
    >>> route.model
    'gpt-4o'
"""

# region imports
from __future__ import annotations

import os
from collections.abc import Callable

import httpx
from pydantic import BaseModel, ConfigDict

from ..config import get_config, is_config_initialized
from ..exceptions import IrisConfigError, IrisProviderError, IrisValidationError
from ..providers import (
    AnthropicMessageAdapter,
    OpenAIMessageAdapter,
    ProviderAdapter,
    ProviderClient,
)

# endregion


_ADAPTER_REGISTRY: dict[str, Callable[[], ProviderAdapter]] = {
    "openai": OpenAIMessageAdapter,
    "anthropic": AnthropicMessageAdapter,
}


class ModelRoute(BaseModel):
    """模型路由解析结果。

    Attributes:
        provider (str): Provider 名称，例如 `"openai"`。
        model (str): 剥离 provider 前缀后的模型名，例如 `"gpt-4o"`。

    Example:
        >>> ModelRoute(provider="openai", model="gpt-4o").provider
        'openai'
    """

    provider: str
    model: str

    model_config = ConfigDict(frozen=True)


def parse_model_route(model: str) -> ModelRoute:
    """解析 `provider/model` 格式的模型字符串。

    Args:
        model (str): 形如 `"openai/gpt-4o"` 的模型字符串。

    Returns:
        ModelRoute: Provider 与剥离前缀后的模型名。

    Raises:
        IrisValidationError: 模型字符串缺少 `/`、provider 为空或模型名为空时抛出。

    Example:
        >>> parse_model_route("openai/gpt-4o").model
        'gpt-4o'
    """
    provider, separator, provider_model = model.partition("/")
    if not separator or not provider or not provider_model:
        raise IrisValidationError("模型字符串必须使用 provider/model 格式", model=model)
    return ModelRoute(provider=provider, model=provider_model)


def create_provider_client(
    model: str | ModelRoute,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    http_client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
) -> ProviderClient:
    """根据模型路由创建 provider client。

    Args:
        model (str | ModelRoute): 模型路由字符串或已解析的路由对象。
        api_key (str | None): 显式 API key，优先级最高。
        base_url (str | None): 自定义 provider base URL。
        timeout (float | None): 请求超时时间，单位秒。
        http_client (httpx.AsyncClient | None): 可注入的 HTTP client。
        headers (dict[str, str] | None): 追加或覆盖的 HTTP headers。

    Returns:
        ProviderClient: 已注入对应 adapter 和 API key 的 provider client。

    Raises:
        IrisProviderError: provider 尚未注册时抛出。
        IrisConfigError: 无法解析 API key 时抛出。

    Example:
        >>> create_provider_client("openai/gpt-4o", api_key="test").adapter.provider
        'openai'
    """
    route = model if isinstance(model, ModelRoute) else parse_model_route(model)
    adapter_factory = _adapter_factory_for(route.provider)
    return ProviderClient(
        adapter=adapter_factory(),
        api_key=_resolve_api_key(route.provider, api_key),
        base_url=base_url,
        timeout=timeout,
        http_client=http_client,
        headers=headers or {},
    )


def _adapter_factory_for(provider: str) -> Callable[[], ProviderAdapter]:
    """返回 provider 对应的 adapter 工厂。"""
    try:
        return _ADAPTER_REGISTRY[provider]
    except KeyError as exc:
        raise IrisProviderError("不支持的 provider", provider=provider) from exc


def _resolve_api_key(provider: str, explicit_api_key: str | None) -> str:
    """按优先级解析 provider API key。"""
    if explicit_api_key:
        return explicit_api_key

    env_var = f"IRIS_{provider.upper()}_API_KEY"
    env_api_key = os.getenv(env_var)
    if env_api_key:
        return env_api_key

    if is_config_initialized():
        config_api_key = get_config().api_key
        if config_api_key:
            return config_api_key

    raise IrisConfigError("缺少 provider API key", provider=provider, env_var=env_var)


__all__ = ["ModelRoute", "create_provider_client", "parse_model_route"]
