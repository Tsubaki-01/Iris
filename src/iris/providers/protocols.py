"""供模型调用方共用的完整响应与可选流式协议。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ..message import LLMRequest, LLMResponse, ModelStreamEvent


class CompletionProvider(Protocol):
    """接收 provider-neutral 请求并支持完整输入 token 估算。"""

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """估算请求中的实际消息、工具和请求选项，不发起生成。"""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """执行一次非流式模型请求并返回标准响应。"""


@runtime_checkable
class StreamingProvider(Protocol):
    """独立于 runtime 的可选 typed streaming 能力。"""

    def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """直接返回有序事件迭代器，调用方不 await 建流结果。"""


def streaming_provider_for(provider: CompletionProvider) -> StreamingProvider | None:
    """返回当前 provider 的可选流式能力，缺失时返回 None。"""
    if isinstance(provider, StreamingProvider):
        return provider
    return None


__all__ = ["CompletionProvider", "StreamingProvider", "streaming_provider_for"]
