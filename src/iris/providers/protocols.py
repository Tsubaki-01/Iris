"""提供 runtime 与显式摘要共同使用的非流式模型调用合同。"""

from __future__ import annotations

from typing import Protocol

from ..message import LLMRequest, LLMResponse


class CompletionProvider(Protocol):
    """接收 provider-neutral 请求并支持完整输入 token 估算。"""

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """估算请求中的实际消息、工具和请求选项，不发起生成。"""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """执行一次非流式模型请求并返回标准响应。"""


__all__ = ["CompletionProvider"]
