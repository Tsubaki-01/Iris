"""模型包装的能力、结果与已知用量契约。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    Decision,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.types import Attributes

from iris.exceptions import IrisProviderError
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.observability.config import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.provider import observe_provider
from iris.observability.service import Observability
from iris.providers.protocols import streaming_provider_for

pytestmark = pytest.mark.usefixtures("otel_test_environment")


@pytest.fixture
def telemetry() -> Iterator[tuple[Observability, InMemorySpanExporter, TracerProvider]]:
    provider = TracerProvider(sampler=ALWAYS_ON)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    yield service, exporter, provider
    provider.shutdown()


class CompletionOnly:
    def __init__(self, result: LLMResponse | BaseException) -> None:
        self.result = result
        self.requests: list[LLMRequest] = []
        self.estimates: list[LLMRequest] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        self.estimates.append(request)
        return 19

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_disabled_preserves_identity_before_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe(provider: object) -> None:
        pytest.fail("禁用采集不应探测 streaming 能力")

    monkeypatch.setattr("iris.observability.provider.streaming_provider_for", unexpected_probe)
    raw = CompletionOnly(LLMResponse(provider="fake"))
    service = Observability.from_config(AgentObservabilityConfig(), ObservabilityExportConfig())
    assert observe_provider(raw, service) is raw


@pytest.mark.asyncio
async def test_complete_preserves_capability_identity_and_one_model_span(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, provider = telemetry
    response = LLMResponse(
        provider="actual-provider",
        id="response-1",
        model="actual-model",
        finish_reason="stop",
        content=[TextBlock(text="answer")],
        input_tokens=0,
        output_tokens=7,
    )
    request = LLMRequest(model="route-name", messages=[Msg.user("hello")])
    raw = CompletionOnly(response)
    observed = observe_provider(raw, service)
    assert streaming_provider_for(observed) is None
    assert observed.estimate_input_tokens(request) == 19
    assert raw.estimates == [request]
    assert exporter.get_finished_spans() == ()
    with provider.get_tracer("test").start_as_current_span("host") as host:
        assert await observed.complete(request) is response
        assert trace.get_current_span() is host
    assert raw.requests == [request]
    model, host_span = exporter.get_finished_spans()
    assert model.name == "chat route-name"
    assert model.kind is SpanKind.CLIENT
    assert model.parent.span_id == host_span.context.span_id
    assert model.status.status_code is StatusCode.UNSET
    assert model.attributes["gen_ai.operation.name"] == "chat"
    assert model.attributes["gen_ai.request.model"] == "route-name"
    assert model.attributes["gen_ai.response.model"] == "actual-model"
    assert model.attributes["gen_ai.provider.name"] == "actual-provider"
    assert model.attributes["gen_ai.response.id"] == "response-1"
    assert model.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert model.attributes["gen_ai.usage.input_tokens"] == 0
    assert model.attributes["gen_ai.usage.output_tokens"] == 7
    assert "gen_ai.usage.total_tokens" not in model.attributes
    assert model.attributes["iris.model.outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [{}, {"input_tokens": 0}, {"output_tokens": 4}])
async def test_complete_publishes_only_explicit_counts(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
    usage: dict[str, int],
) -> None:
    service, exporter, _ = telemetry
    response = LLMResponse(provider="fake", **usage)
    await observe_provider(CompletionOnly(response), service).complete(LLMRequest(model="model"))
    attributes = exporter.get_finished_spans()[0].attributes
    for name in ("input_tokens", "output_tokens"):
        assert attributes.get(f"gen_ai.usage.{name}") == usage.get(name)


@pytest.mark.asyncio
async def test_provider_exception_is_preserved_with_sparse_usage(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    error = IrisProviderError("failed", usage={"output_tokens": 0})
    raw = CompletionOnly(error)
    with pytest.raises(IrisProviderError) as caught:
        await observe_provider(raw, service).complete(LLMRequest(model="model"))
    assert caught.value is error
    assert len(raw.requests) == 1
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["iris.model.outcome"] == "failed"
    assert span.attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert "gen_ai.provider.name" not in span.attributes


@pytest.mark.asyncio
async def test_provider_exception_with_no_usage_is_preserved(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    error = IrisProviderError("failed without reported usage", usage=None)
    raw = CompletionOnly(error)
    with pytest.raises(IrisProviderError) as caught:
        await observe_provider(raw, service).complete(LLMRequest(model="test"))
    assert caught.value is error
    [span] = exporter.get_finished_spans()
    assert span.attributes["iris.model.outcome"] == "failed"
    assert "gen_ai.usage.input_tokens" not in span.attributes


@pytest.mark.asyncio
async def test_cancellation_is_preserved_without_provider_error(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    error = asyncio.CancelledError()
    raw = CompletionOnly(error)
    with pytest.raises(asyncio.CancelledError) as caught:
        await observe_provider(raw, service).complete(LLMRequest(model="model"))
    assert caught.value is error
    assert len(raw.requests) == 1
    span = exporter.get_finished_spans()[0]
    assert span.attributes["iris.model.outcome"] == "cancelled"
    assert span.status.status_code is StatusCode.UNSET


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["start", "project", "set", "end"])
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_telemetry_failures_never_retry_or_replace_business_outcome(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fault: str,
    outcome: str,
) -> None:
    _, _, provider = telemetry
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    result: LLMResponse | BaseException
    if outcome == "completed":
        result = LLMResponse(provider="fake", content=[TextBlock(text="answer")])
    elif outcome == "failed":
        result = IrisProviderError("original failure")
    else:
        result = asyncio.CancelledError()
    raw = CompletionOnly(result)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"telemetry {fault} failed")

    original_start = service._tracer.start_span

    def start(*args: Any, **kwargs: Any) -> trace.Span:
        span = original_start(*args, **kwargs)
        monkeypatch.setattr(span, "set_attributes" if fault == "set" else "end", fail)
        return span

    if fault == "project":
        monkeypatch.setattr("iris.observability.content.request_attributes", fail)
        monkeypatch.setattr("iris.observability.content.response_attributes", fail)
    else:
        monkeypatch.setattr(service._tracer, "start_span", fail if fault == "start" else start)

    with provider.get_tracer("host").start_as_current_span("host") as host:
        observed = observe_provider(raw, service)
        request = LLMRequest(model="fake", messages=[Msg.user("hello")])
        if isinstance(result, BaseException):
            with pytest.raises(type(result)) as caught:
                await observed.complete(request)
            assert caught.value is result
        else:
            assert await observed.complete(request) is result
        assert trace.get_current_span() is host
    assert raw.requests == [request]
    assert f"telemetry {fault} failed" in caplog.text


@pytest.mark.asyncio
async def test_request_and_response_truncation_keep_both_original_field_names(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    _, exporter, provider = telemetry
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=True, max_content_chars=8),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    response = LLMResponse(provider="fake", content=[TextBlock(text="long response")])
    request = LLMRequest(model="fake", messages=[Msg.user("long request")])
    assert await observe_provider(CompletionOnly(response), service).complete(request) is response
    attributes = exporter.get_finished_spans()[0].attributes
    assert set(attributes["iris.content.truncated_fields"]) == {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
    }
    for field in ("gen_ai.input.messages", "gen_ai.output.messages"):
        assert field not in attributes
        assert len(attributes[f"iris.content.{field}.preview"]) <= 8


@pytest.mark.asyncio
async def test_nonrecording_model_preserves_context_business_and_children(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, exporter, child_provider = telemetry
    provider = TracerProvider(sampler=ALWAYS_OFF)
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )

    def unexpected_projection(*args: Any, **kwargs: Any) -> None:
        pytest.fail("不记录的 span 不应投影正文")

    monkeypatch.setattr("iris.observability.content.request_attributes", unexpected_projection)
    monkeypatch.setattr("iris.observability.content.response_attributes", unexpected_projection)
    parent_contexts: list[trace.SpanContext] = []

    class ChildProvider(CompletionOnly):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            parent = trace.get_current_span()
            assert not parent.is_recording()
            parent_contexts.append(parent.get_span_context())
            with child_provider.get_tracer("child").start_as_current_span("child"):
                return await super().complete(request)

    response = LLMResponse(provider="fake")
    raw = ChildProvider(response)
    try:
        assert await observe_provider(raw, service).complete(LLMRequest(model="fake")) is response
        child = exporter.get_finished_spans()[0]
        assert child.parent == parent_contexts[0]
        assert child.parent.is_valid
        assert len(raw.requests) == 1
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_model_operation_is_available_to_the_sampler_at_creation() -> None:
    class ModelSampler(Sampler):
        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Attributes = None,
            links: Sequence[trace.Link] | None = None,
            trace_state: trace.TraceState | None = None,
        ) -> SamplingResult:
            decision = (
                Decision.RECORD_AND_SAMPLE
                if attributes and attributes.get("gen_ai.operation.name") == "chat"
                else Decision.DROP
            )
            return SamplingResult(decision, attributes=attributes)

        def get_description(self) -> str:
            return "model calls"

    provider = TracerProvider(sampler=ModelSampler())
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    try:
        await observe_provider(CompletionOnly(LLMResponse(provider="fake")), service).complete(
            LLMRequest(model="requested")
        )
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.operation.name"] == "chat"
        assert span.attributes["gen_ai.request.model"] == "requested"
    finally:
        provider.shutdown()
