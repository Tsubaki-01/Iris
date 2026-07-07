"""Iris provider 层公共导出。

Provider 包导出 provider client 与 provider client 工厂。LLM 请求与响应模型统一从
`iris.message` 或 `iris.message.llm` 导入，避免 provider 层形成重复模型边界。

Example:
    >>> from iris.providers import ProviderClient
    >>> ProviderClient(provider="openai", api_key="test").provider
    'openai'
"""

# region imports
from .client import ProviderClient
from .factory import ModelRoute, create_provider_client, parse_model_route

# endregion

__all__ = [
    "ModelRoute",
    "ProviderClient",
    "create_provider_client",
    "parse_model_route",
]
