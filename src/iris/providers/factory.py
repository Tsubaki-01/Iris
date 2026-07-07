"""Provider client 工厂。

本模块负责解析高层模型路由，并按内置 provider 与 `Config.providers`
组成的注册表创建对应的 `ProviderClient`。

Example:
    >>> route = parse_model_route("openai/gpt-4o")
    >>> route.model
    'gpt-4o'
"""

# region imports
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..config import ProviderConfig, get_config, is_config_initialized
from ..exceptions import IrisConfigError, IrisProviderError, IrisValidationError
from .client import ProviderClient

# endregion

BUILTIN_PROVIDER_IDS = frozenset({"openai", "anthropic", "deepseek"})


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
    headers: dict[str, str] | None = None,
) -> ProviderClient:
    """根据模型路由创建 provider client。

    Args:
        model (str | ModelRoute): 模型路由字符串或已解析的路由对象。
        api_key (str | None): 显式 API key，优先级最高。
        base_url (str | None): 自定义 provider base URL。
        timeout (float | None): 请求超时时间，单位秒。
        headers (dict[str, str] | None): 追加或覆盖的 HTTP headers。

    Returns:
        ProviderClient: 已注入 provider 和 API key 的 provider client。

    Raises:
        IrisProviderError: provider 尚未注册时抛出。
        IrisConfigError: 无法解析 API key 时抛出。

    Example:
        >>> create_provider_client("openai/gpt-4o", api_key="test").provider
        'openai'
    """
    route = model if isinstance(model, ModelRoute) else parse_model_route(model)
    provider_config = _resolve_provider_config(route.provider)
    return ProviderClient(
        provider=route.provider,
        litellm_provider=provider_config.litellm_provider,
        api_key=_resolve_api_key(route.provider, api_key),
        base_url=base_url or provider_config.base_url,
        timeout=timeout,
        headers={**provider_config.headers, **(headers or {})},
    )


def _resolve_provider_config(provider: str) -> ProviderConfig:
    """从内置和用户声明的注册表中解析 provider 配置。"""
    provider_config = _provider_registry().get(provider)
    if provider_config is None:
        raise IrisProviderError("未注册 provider", provider=provider)
    return provider_config


def _provider_registry() -> dict[str, ProviderConfig]:
    """返回当前可用 provider 注册表。

    内置服务商默认配置 + 用户自定义配置合并，支持自定义第三方服务商。"""
    registry = {
        provider: ProviderConfig(litellm_provider=provider) for provider in BUILTIN_PROVIDER_IDS
    }
    if not is_config_initialized():
        return registry

    for provider, provider_config in get_config().providers.items():
        if provider in BUILTIN_PROVIDER_IDS:
            registry[provider] = _merge_builtin_provider_config(
                provider,
                provider_config,
            )
        elif provider_config.base_url:
            registry[provider] = provider_config
    return registry


def _merge_builtin_provider_config(
    provider: str,
    override: ProviderConfig,
) -> ProviderConfig:
    """合并内置 provider 默认配置与用户非 secret override。

    如果用户显式配置了 litellm_provider，尊重用户配置。
    如果用户没显式配置，只是 ProviderConfig 自带默认 "openai"，那内置 provider 继续用自己的 provider
    """
    litellm_provider = (
        override.litellm_provider if "litellm_provider" in override.model_fields_set else provider
    )
    return ProviderConfig(
        litellm_provider=litellm_provider,
        base_url=override.base_url,
        api_style=override.api_style,
        headers=override.headers,
    )


def _resolve_api_key(provider: str, explicit_api_key: str | None) -> str:
    """按优先级解析 provider API key。"""
    if explicit_api_key:
        return explicit_api_key

    if is_config_initialized():
        config = get_config()
        provider_api_key = config.provider_api_keys.get(provider)
        if provider_api_key:
            return provider_api_key
        if config.api_key:
            return config.api_key

    raise IrisConfigError("缺少 provider API key", provider=provider)


__all__ = [
    "BUILTIN_PROVIDER_IDS",
    "ModelRoute",
    "create_provider_client",
    "parse_model_route",
]
