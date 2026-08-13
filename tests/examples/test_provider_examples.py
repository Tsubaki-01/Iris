from __future__ import annotations

import pytest

from examples.provider.basic import build_request, complete_once
from examples.provider.trace import TracingProvider
from iris.exceptions import IrisProviderError
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.providers import create_provider_client


def _response(text: str = "完成") -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake-model",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
    )


class StaticProvider:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


class FailingProvider:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        raise IrisProviderError("调用失败", provider="fake")


def test_build_request_uses_provider_internal_model() -> None:
    request = build_request(model="deepseek-chat", prompt="介绍 Iris")
    assert request.model == "deepseek-chat"
    assert [message.role.value for message in request.messages] == ["system", "user"]
    assert [message.text for message in request.messages] == [
        "你是一个简洁的助手。",
        "介绍 Iris",
    ]


def test_current_provider_factory_constructs_without_legacy_adapter() -> None:
    client = create_provider_client("deepseek/deepseek-chat", api_key="test-key")
    assert client.provider == "deepseek"
    assert client.api_key == "test-key"


@pytest.mark.asyncio
async def test_complete_once_uses_injected_runtime_provider() -> None:
    provider = StaticProvider(_response("你好"))
    request = build_request(model="fake-model", prompt="问题")
    response = await complete_once(provider, request)
    assert response.to_msg().text == "你好"
    assert provider.requests == [request]


@pytest.mark.asyncio
async def test_tracing_provider_records_response_and_error() -> None:
    request = LLMRequest(model="fake-model", messages=[Msg.user("问题")])
    traced = TracingProvider(StaticProvider(_response("成功")))
    assert (await traced.complete(request)).to_msg().text == "成功"
    assert traced.records[0].snapshot()["response"] is not None

    failing = TracingProvider(FailingProvider())
    with pytest.raises(IrisProviderError):
        await failing.complete(request)
    assert "IrisProviderError" in (failing.records[0].error or "")
