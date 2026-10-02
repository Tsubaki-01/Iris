"""模型提供者调用与流式协议异常。"""

from .base import IrisError


class IrisProviderError(IrisError):
    """模型提供者和 LLM 集成错误的基类。"""

    runtime_error_source = "provider"
    runtime_error_code = "PROVIDER_ERROR"


class IrisProviderStreamError(IrisProviderError):
    """Provider 流式调用失败的基类。"""

    runtime_error_code = "PROVIDER_STREAM_ERROR"


class IrisProviderStreamProtocolError(IrisProviderStreamError):
    """Provider 流式响应违反 chunk 或终态协议。"""

    runtime_error_code = "PROVIDER_STREAM_PROTOCOL_ERROR"


class IrisProviderStreamInterruptedError(IrisProviderStreamError):
    """Provider 流在合法终态前意外结束。"""

    runtime_error_code = "PROVIDER_STREAM_INTERRUPTED"


class IrisAPIConnectionError(IrisProviderError):
    """连接到提供者 API 失败时抛出。"""


class IrisRateLimitExceededError(IrisProviderError):
    """超出提供者 API 速率限制时抛出。"""


class IrisAuthenticationError(IrisProviderError):
    """提供者 API 身份认证失败时抛出。"""
