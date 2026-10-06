"""流式采集的终态、消费区间与上下文隔离。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from iris.exceptions import IrisProviderError
from iris.message import (
    LLMRequest,
    LLMResponse,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ModelUsageSnapshot,
    ModelUsageUpdated,
    ProviderStreamError,
)
from iris.observability.config import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.provider import observe_provider
from iris.observability.service import Observability
from iris.providers.protocols import streaming_provider_for


@pytest.fixture
def telemetry() -> Iterator[tuple[Observability, InMemorySpanExporter, TracerProvider]]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    yield service, exporter, provider
    provider.shutdown()


_SCOPE = ModelStreamScope(model_stream_id="stream-1", provider="fake", model="actual", attempt=1)
_TIME = datetime.now(UTC)


def _started() -> ModelResponseStarted:
    return ModelResponseStarted(
        scope=_SCOPE, sequence=1, occurred_at=_TIME, response_id="response-1"
    )


def _completed() -> ModelResponseCompleted:
    return ModelResponseCompleted(
        scope=_SCOPE,
        sequence=2,
        occurred_at=_TIME,
        response=LLMResponse(provider="fake", id="response-1", model="actual"),
        semantic_output_emitted=False,
    )


class EventsIterator:
    """真实 AsyncIterator 可以不提供 aclose。"""

    def __init__(self, events: list[ModelStreamEvent | BaseException]) -> None:
        self.events = iter(events)
        self.pulls = 0
        self.parents: list[int] = []

    def __aiter__(self) -> AsyncIterator[ModelStreamEvent]:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        self.pulls += 1
        self.parents.append(trace.get_current_span().get_span_context().span_id)
        event = next(self.events, None)
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, BaseException):
            raise event
        return event


class ClosingIterator(EventsIterator):
    def __init__(
        self,
        events: list[ModelStreamEvent | BaseException],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        super().__init__(events)
        self.closed = False
        self.close_error = close_error
        self.close_span = None

    async def aclose(self) -> None:
        self.closed = True
        self.close_span = trace.get_current_span()
        if self.close_error is not None:
            raise self.close_error


class StreamingFake:
    def __init__(
        self,
        iterator: EventsIterator,
        *,
        init_error: BaseException | None = None,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.iterator = iterator
        self.init_error = init_error
        self.tracer = tracer
        self.requests: list[LLMRequest] = []
        self.init_parent: int | None = None

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return 12

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(provider="fake")

    def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self.init_parent = trace.get_current_span().get_span_context().span_id
        if self.tracer is not None:
            with self.tracer.start_as_current_span("synchronous-init"):
                pass
        if self.init_error is not None:
            raise self.init_error
        return self.iterator


def _stream(raw: StreamingFake, service: Observability) -> AsyncIterator[ModelStreamEvent]:
    provider = streaming_provider_for(observe_provider(raw, service))
    assert provider is not None
    return provider.stream(LLMRequest(model="requested", stream=True))


@pytest.mark.asyncio
async def test_unconsumed_stream_does_not_call_raw_or_create_span(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    raw = StreamingFake(EventsIterator([_completed()]))
    events = _stream(raw, service)
    assert raw.requests == []
    assert exporter.get_finished_spans() == ()
    await events.aclose()
    assert raw.requests == []
    assert exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_creation_context_and_each_pull_are_isolated_across_tasks(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, provider = telemetry
    tracer = provider.get_tracer("test")
    terminal = _completed()
    iterator = ClosingIterator([_started(), terminal])
    raw = StreamingFake(iterator, tracer=tracer)
    with tracer.start_as_current_span("creation") as creation:
        with service.bind({"iris.model.purpose": "compaction", "iris.run.id": "creation-run"}):
            events = _stream(raw, service)

    async def pull(name: str) -> ModelStreamEvent:
        with tracer.start_as_current_span(name) as consumer:
            event = await anext(events)
            assert trace.get_current_span() is consumer
            return event

    assert await asyncio.create_task(pull("consumer-1")) is not terminal
    assert await asyncio.create_task(pull("consumer-2")) is terminal
    with tracer.start_as_current_span("cleanup") as cleanup:
        await events.aclose()
        assert trace.get_current_span() is cleanup
    spans = {span.name: span for span in exporter.get_finished_spans()}
    model = spans["chat requested"]
    assert model.parent.span_id == creation.get_span_context().span_id
    assert spans["synchronous-init"].parent.span_id == model.context.span_id
    assert raw.init_parent == model.context.span_id
    assert iterator.parents == [model.context.span_id, model.context.span_id]
    assert iterator.closed
    assert iterator.close_span.get_span_context().span_id == model.context.span_id
    assert model.attributes["iris.model.outcome"] == "completed"
    assert model.attributes["gen_ai.provider.name"] == "fake"
    assert model.attributes["gen_ai.response.model"] == "actual"
    assert model.attributes["iris.model.purpose"] == "compaction"
    assert model.attributes["iris.run.id"] == "creation-run"


@pytest.mark.asyncio
async def test_iterator_without_aclose_preserves_events_and_terminal(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    events = [_started(), _completed()]
    iterator = EventsIterator(events)
    result = [event async for event in _stream(StreamingFake(iterator), service)]
    assert result == events
    assert all(actual is expected for actual, expected in zip(result, events, strict=True))
    assert exporter.get_finished_spans()[0].attributes["iris.model.outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["failed", "cancelled"])
async def test_typed_terminal_is_fixed_before_immediate_close(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider], kind: str
) -> None:
    service, exporter, _ = telemetry
    error = ProviderStreamError(code="PROVIDER_ERROR", message="failed", retryable=False)
    terminal: ModelStreamEvent
    if kind == "failed":
        terminal = ModelResponseFailed(
            scope=_SCOPE,
            sequence=1,
            occurred_at=_TIME,
            error=error,
            semantic_output_emitted=False,
        )
    else:
        terminal = ModelResponseCancelled(
            scope=_SCOPE,
            sequence=1,
            occurred_at=_TIME,
            error=error,
            semantic_output_emitted=False,
        )
    iterator = ClosingIterator([terminal])
    events = _stream(StreamingFake(iterator), service)
    assert await anext(events) is terminal
    assert exporter.get_finished_spans() == ()
    await events.aclose()
    span = exporter.get_finished_spans()[0]
    assert span.attributes["iris.model.outcome"] == kind
    assert span.status.status_code is (StatusCode.ERROR if kind == "failed" else StatusCode.UNSET)
    assert iterator.closed
    assert iterator.pulls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", [False, True])
async def test_missing_terminal_distinguishes_eof_and_abandonment(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider], close_early: bool
) -> None:
    service, exporter, _ = telemetry
    iterator = ClosingIterator([_started()])
    events = _stream(StreamingFake(iterator), service)
    await anext(events)
    if close_early:
        await events.aclose()
    else:
        with pytest.raises(StopAsyncIteration):
            await anext(events)
    span = exporter.get_finished_spans()[0]
    assert span.attributes["iris.model.outcome"] == ("abandoned" if close_early else "failed")
    assert span.status.status_code is (StatusCode.UNSET if close_early else StatusCode.ERROR)
    assert iterator.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("during_init", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_raw_failure_and_cancellation_propagate_and_close_once(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
    during_init: bool,
    cancelled: bool,
) -> None:
    service, exporter, _ = telemetry
    error = asyncio.CancelledError() if cancelled else IrisProviderError("raw failure")
    iterator = ClosingIterator([error])
    raw = StreamingFake(iterator, init_error=error if during_init else None)
    events = _stream(raw, service)
    with pytest.raises(type(error)) as caught:
        await anext(events)
    assert caught.value is error
    assert len(raw.requests) == 1
    assert iterator.closed is (not during_init)
    span = exporter.get_finished_spans()[0]
    assert span.attributes["iris.model.outcome"] == ("cancelled" if cancelled else "failed")
    assert span.status.status_code is (StatusCode.UNSET if cancelled else StatusCode.ERROR)


@pytest.mark.asyncio
async def test_latest_usage_snapshot_replaces_prior_snapshot_without_summing(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    snapshots = [
        ModelUsageSnapshot(input_tokens=8, output_tokens=2),
        ModelUsageSnapshot(output_tokens=0, complete=True),
    ]
    iterator = ClosingIterator(
        [
            ModelUsageUpdated(scope=_SCOPE, sequence=i, occurred_at=_TIME, usage=usage)
            for i, usage in enumerate(snapshots, 1)
        ]
    )
    events = _stream(StreamingFake(iterator), service)
    await anext(events)
    await anext(events)
    await events.aclose()
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.total_tokens" not in attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_received", [False, True])
async def test_consumer_exception_and_close_error_preserve_business_outcome(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
    caplog: pytest.LogCaptureFixture,
    terminal_received: bool,
) -> None:
    service, exporter, _ = telemetry
    event = _completed() if terminal_received else _started()
    iterator = ClosingIterator([event], close_error=IrisProviderError("close failed"))
    events = _stream(StreamingFake(iterator), service)
    original = RuntimeError("consumer failed")
    with pytest.raises(RuntimeError) as caught:
        try:
            await anext(events)
            raise original
        finally:
            await events.aclose()
    assert caught.value is original
    assert iterator.pulls == 1
    span = exporter.get_finished_spans()[0]
    assert span.attributes["iris.model.outcome"] == (
        "completed" if terminal_received else "abandoned"
    )
    assert span.status.status_code is StatusCode.UNSET
    assert "close failed" in caplog.text


@pytest.mark.asyncio
async def test_async_generator_provider_keeps_complete_and_estimate_and_sparse_final_usage(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    response = LLMResponse(provider="fake", output_tokens=3)
    terminal = _completed().model_copy(update={"response": response})

    class AsyncGeneratorProvider(StreamingFake):
        async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield terminal

    raw = AsyncGeneratorProvider(EventsIterator([]))
    observed = observe_provider(raw, service)
    request = LLMRequest(model="requested")
    assert observed.estimate_input_tokens(request) == 12
    assert (await observed.complete(request)).provider == "fake"
    events = _stream(raw, service)
    assert await anext(events) is terminal
    await events.aclose()
    assert raw.requests == [request.model_copy(update={"stream": True})]
    complete, streamed = exporter.get_finished_spans()
    assert complete.attributes["iris.model.outcome"] == "completed"
    assert streamed.attributes["iris.model.outcome"] == "completed"
    assert streamed.attributes["gen_ai.usage.output_tokens"] == 3
    assert "gen_ai.usage.input_tokens" not in streamed.attributes


@pytest.mark.asyncio
async def test_cancelled_cleanup_propagates_and_finishes_span(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, provider = telemetry
    cancellation = asyncio.CancelledError()
    iterator = ClosingIterator([_started()], close_error=cancellation)
    events = _stream(StreamingFake(iterator), service)
    await anext(events)
    with provider.get_tracer("test").start_as_current_span("consumer") as consumer:
        with pytest.raises(asyncio.CancelledError) as caught:
            await events.aclose()
        assert caught.value is cancellation
        assert trace.get_current_span() is consumer
    assert exporter.get_finished_spans()[0].attributes["iris.model.outcome"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_usage", [{}, {"input_tokens": 2, "output_tokens": 1}])
async def test_latest_snapshot_remains_authoritative_after_completed_response(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
    terminal_usage: dict[str, int],
) -> None:
    service, exporter, _ = telemetry
    snapshot = ModelUsageUpdated(
        scope=_SCOPE,
        sequence=1,
        occurred_at=_TIME,
        usage=ModelUsageSnapshot(input_tokens=17, output_tokens=9),
    )
    terminal = _completed().model_copy(
        update={"response": LLMResponse(provider="fake", **terminal_usage)}
    )
    events = _stream(StreamingFake(EventsIterator([snapshot, terminal])), service)
    assert [event async for event in events] == [snapshot, terminal]
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["gen_ai.usage.input_tokens"] == 17
    assert attributes["gen_ai.usage.output_tokens"] == 9


@pytest.mark.asyncio
async def test_stream_exception_usage_is_sparse_and_preserved(
    telemetry: tuple[Observability, InMemorySpanExporter, TracerProvider],
) -> None:
    service, exporter, _ = telemetry
    error = IrisProviderError("failed stream", usage={"output_tokens": 0})
    events = _stream(StreamingFake(EventsIterator([error])), service)
    with pytest.raises(IrisProviderError) as caught:
        await anext(events)
    assert caught.value is error
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.input_tokens" not in attributes
