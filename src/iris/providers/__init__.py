"""Iris provider 层公共导出。

Provider 包导出适配器、HTTP client 与 provider client 工厂。LLM 请求与响应模型统一从
`iris.message` 或 `iris.message.llm` 导入，避免 provider 层形成重复模型边界。

Example:
    >>> from iris.providers import OpenAIMessageAdapter
    >>> OpenAIMessageAdapter().provider
    'openai'
"""

# region imports
from .adapter import ProviderAdapter
from .anthropic import AnthropicMessageAdapter
from .client import ProviderClient
from .factory import ModelRoute, create_provider_client, parse_model_route
from .openai import OpenAIMessageAdapter

# endregion

__all__ = [
    "AnthropicMessageAdapter",
    "ModelRoute",
    "OpenAIMessageAdapter",
    "ProviderAdapter",
    "ProviderClient",
    "create_provider_client",
    "parse_model_route",
]
