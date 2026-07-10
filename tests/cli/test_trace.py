from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from iris.cli.trace import ChatTraceStore, TracingRuntimeProvider
from iris.exceptions import IrisProviderError
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.runtime import RuntimeProvider


class FakeProvider:
    """CLI 测试用 provider。"""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self.responses = list(responses)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """返回下一条响应。"""
        del request
        if not self.responses:
            raise IrisProviderError("没有可用响应", provider="fake")
        return self.responses.pop(0)


class FailingProvider:
    """始终失败的 provider。"""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """抛出 provider 错误。"""
        del request
        raise IrisProviderError("调用失败", provider="fake")


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake-model",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
    )


@pytest.mark.asyncio
async def test_tracing_provider_records_request_response_and_jsonl(tmp_path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    store = ChatTraceStore(trace_file)
    store.start_turn(1)
    provider: RuntimeProvider = TracingRuntimeProvider(
        FakeProvider([_response("你好")]),
        store,
    )
    request = LLMRequest(model="fake-model", messages=[Msg.user("问题")])

    response = await provider.complete(request)

    assert response.to_msg().text == "你好"
    steps = store.steps_for_turn(1)
    assert len(steps) == 1
    assert steps[0].request == request
    assert steps[0].response == response
    assert steps[0].error is None

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["turn_index"] == 1
    assert payload["request"]["model"] == "fake-model"
    assert payload["response"]["provider"] == "fake"


@pytest.mark.asyncio
async def test_tracing_provider_records_error_then_reraises() -> None:
    store = ChatTraceStore()
    store.start_turn(2)
    provider: RuntimeProvider = TracingRuntimeProvider(FailingProvider(), store)

    with pytest.raises(IrisProviderError):
        await provider.complete(LLMRequest(model="fake-model", messages=[]))

    steps = store.steps_for_turn(2)
    assert len(steps) == 1
    assert steps[0].response is None
    assert "IrisProviderError" in (steps[0].error or "")
