"""Iris provider 层公共导出。

Provider 包只导出适配器与 HTTP client。LLM 请求与响应模型统一从
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
from .openai import OpenAIMessageAdapter

# endregion

__all__ = [
    "AnthropicMessageAdapter",
    "OpenAIMessageAdapter",
    "ProviderAdapter",
    "ProviderClient",
]
