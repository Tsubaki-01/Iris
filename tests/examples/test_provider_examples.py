from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from io import StringIO

import pytest

from examples.provider.basic import build_request, stream_once
from examples.provider.trace import TracingProvider
from iris.exceptions import IrisProviderStreamError
from iris.message import (
    LLMRequest,
    LLMResponse,
    ModelBlockDelta,
    ModelBlockRef,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelStreamEvent,
    ModelStreamScope,
    Msg,
    ProviderStreamError,
    TextBlock,
)
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


def _stream_events(response: LLMResponse) -> list[ModelStreamEvent]:
    scope = ModelStreamScope(
        model_stream_id="stream-1",
        provider=response.provider,
        model=response.model,
        attempt=1,
    )
    block = ModelBlockRef(index=0, block_id="text-0", kind="text")
    occurred_at = datetime.now(UTC)
    return [
        ModelBlockDelta(
            scope=scope,
            sequence=1,
            occurred_at=occurred_at,
            block=block,
            channel="text",
            delta="你",
            snapshot="你",
        ),
        ModelBlockDelta(
            scope=scope,
            sequence=2,
            occurred_at=occurred_at,
            block=block,
            channel="text",
            delta="好",
            snapshot="你好",
        ),
        ModelResponseCompleted(
            scope=scope,
            sequence=3,
            occurred_at=occurred_at,
            response=response,
            semantic_output_emitted=True,
        ),
    ]


class StreamingProvider:
    def __init__(self, events: Sequence[ModelStreamEvent]) -> None:
        self.events = events
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event


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
async def test_stream_once_writes_text_deltas_and_returns_completed_response() -> None:
    provider = StreamingProvider(_stream_events(_response("你好")))
    request = build_request(model="fake-model", prompt="问题")
    output = StringIO()

    response = await stream_once(provider, request, output=output)

    assert output.getvalue() == "你好"
    assert response.to_msg().text == "你好"
    assert provider.requests == [request.model_copy(update={"stream": True})]


@pytest.mark.asyncio
async def test_stream_once_raises_safe_provider_terminal_error() -> None:
    response = _response()
    completed = _stream_events(response)[-1]
    assert isinstance(completed, ModelResponseCompleted)
    failed = ModelResponseFailed(
        scope=completed.scope,
        sequence=1,
        occurred_at=completed.occurred_at,
        error=ProviderStreamError(
            code="PROVIDER_STREAM_ERROR",
            message="调用失败",
            retryable=True,
        ),
        semantic_output_emitted=False,
    )

    with pytest.raises(IrisProviderStreamError, match="调用失败"):
        await stream_once(
            StreamingProvider([failed]),
            build_request(model="fake-model", prompt="问题"),
            output=StringIO(),
        )


@pytest.mark.asyncio
async def test_tracing_provider_records_stream_response_and_safe_error() -> None:
    request = LLMRequest(model="fake-model", messages=[Msg.user("问题")])
    traced = TracingProvider(StreamingProvider(_stream_events(_response("成功"))))

    events = [event async for event in traced.stream(request)]

    assert isinstance(events[-1], ModelResponseCompleted)
    assert traced.records[0].snapshot()["response"] is not None

    failed = ModelResponseFailed(
        scope=events[-1].scope,
        sequence=1,
        occurred_at=events[-1].occurred_at,
        error=ProviderStreamError(
            code="PROVIDER_STREAM_ERROR",
            message="调用失败",
            retryable=True,
        ),
        semantic_output_emitted=False,
    )
    failing = TracingProvider(StreamingProvider([failed]))

    emitted = [event async for event in failing.stream(request)]

    assert emitted == [failed]
    assert failing.records[0].error == "PROVIDER_STREAM_ERROR: 调用失败"
