from __future__ import annotations

from collections.abc import Sequence

from iris.exceptions import IrisProviderError
from iris.message import LLMRequest, LLMResponse


class FakeProvider:
    """测试用 provider，只记录请求并按顺序返回预设响应。"""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self._requests: list[LLMRequest] = []

    @property
    def requests(self) -> list[LLMRequest]:
        """返回已捕获的请求快照。"""
        return list(self._requests)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并返回下一条预设响应。"""
        self._requests.append(request)
        if not self._responses:
            raise IrisProviderError("FakeProvider 响应已耗尽", provider="fake")
        return self._responses.pop(0)
