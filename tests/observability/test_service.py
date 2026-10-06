"""标准 OTel 服务的关闭责任、上下文与记录失败隔离。"""

import asyncio
import threading
from contextvars import ContextVar
from unittest.mock import Mock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import INVALID_SPAN, StatusCode

from iris.exceptions import IrisCancellationRequestedError
from iris.message import LLMRequest
from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability


def service(provider: TracerProvider, *, capture: bool = False) -> Observability:
    return Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=capture),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )


def test_scope_restores_context_and_records_business_exception() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    obs = service(provider)
    original = trace.get_current_span()
    failure = RuntimeError("business")
    with pytest.raises(RuntimeError) as caught, obs.bind({"iris.run.id": "run-1"}):
        with obs.scope("work") as span:
            assert trace.get_current_span() is span
            raise failure
    assert caught.value is failure
    assert trace.get_current_span() is original
    saved = exporter.get_finished_spans()[0]
    assert saved.attributes["iris.run.id"] == "run-1"
    assert saved.status.status_code == StatusCode.ERROR
    provider.shutdown()


def test_disabled_does_not_change_host_context() -> None:
    provider = TracerProvider()
    obs = Observability.from_config(AgentObservabilityConfig(), ObservabilityExportConfig())
    with provider.get_tracer("host").start_as_current_span("host") as host:
        with obs.scope("off") as span, obs.bind({"iris.run.id": "off"}), obs.detached():
            assert span is INVALID_SPAN
            assert trace.get_current_span() is host
    provider.shutdown()


def test_detached_clears_only_otel_and_bind_replaces_identity() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    obs = service(provider)
    business: ContextVar[str] = ContextVar("business", default="kept")
    with obs.bind({"iris.run.id": "parent", "iris.step.index": 3}), obs.scope("parent"):
        with obs.bind({"iris.run.id": "child"}, replace=True), obs.scope("child"):
            pass
        with obs.detached(), obs.scope("background"):
            assert business.get() == "kept"
    saved = {span.name: span for span in exporter.get_finished_spans()}
    assert saved["child"].attributes == {"iris.run.id": "child"}
    assert saved["child"].parent.span_id == saved["parent"].context.span_id
    assert saved["background"].parent is None
    assert not saved["background"].attributes
    provider.shutdown()


@pytest.mark.asyncio
async def test_borrowed_provider_is_never_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TracerProvider()
    obs = service(provider)
    shutdown = Mock()
    monkeypatch.setattr(provider, "shutdown", shutdown)
    await obs.aclose()
    shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_owned_provider_shutdown_runs_off_loop_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    monkeypatch.setattr(OTLPSpanExporter, "export", Mock())
    before = trace.get_tracer_provider()
    obs = Observability.from_config(
        AgentObservabilityConfig(enabled=True),
        ObservabilityExportConfig(traces_endpoint="http://localhost:5000/v1/traces"),
    )
    assert trace.get_tracer_provider() is before
    sdk = obs._owned_provider
    original = sdk.shutdown
    threads: list[int] = []

    def shutdown() -> None:
        threads.append(threading.get_ident())
        original()

    monkeypatch.setattr(sdk, "shutdown", shutdown)
    await obs.aclose()
    await obs.aclose()
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_shutdown_failure_keeps_original_business_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TracerProvider(shutdown_on_exit=False)
    original_shutdown = provider.shutdown
    obs = Observability(
        AgentObservabilityConfig(enabled=True),
        provider,
        owned_provider=provider,
    )
    shutdown = Mock(side_effect=RuntimeError("export failure"))
    monkeypatch.setattr(provider, "shutdown", shutdown)
    business = RuntimeError("business failure")
    with pytest.raises(RuntimeError) as caught:
        try:
            raise business
        finally:
            await obs.aclose()
    assert caught.value is business
    await obs.aclose()
    shutdown.assert_called_once()
    original_shutdown()


@pytest.mark.parametrize("operation", ["start", "set", "end", "project"])
def test_observation_failures_do_not_escape_or_leak_context(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris.observability import content

    provider = TracerProvider()
    obs = service(provider, capture=True)
    broken = Mock(side_effect=RuntimeError("telemetry"))
    original_context = trace.get_current_span()
    if operation == "start":
        monkeypatch.setattr(obs._tracer, "start_span", broken)
    if operation == "project":
        monkeypatch.setattr(content, "request_attributes", broken)
    calls = 0
    with obs.scope("work") as span:
        if operation == "set":
            monkeypatch.setattr(span, "set_attributes", broken)
        if operation == "end":
            monkeypatch.setattr(span, "end", broken)
        obs.attributes(span, {"key": "value"})
        obs.record_request(span, LLMRequest(model="test"))
        calls += 1
    assert calls == 1
    assert trace.get_current_span() is original_context
    assert broken.called
    provider.shutdown()


@pytest.mark.parametrize("capture,recording", [(False, True), (True, False)])
def test_content_gate_skips_projection(
    capture: bool,
    recording: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris.observability import content

    provider = TracerProvider() if recording else TracerProvider(sampler=ALWAYS_OFF)
    obs = service(provider, capture=capture)
    project = Mock(side_effect=AssertionError("must not project"))
    monkeypatch.setattr(content, "request_attributes", project)
    with obs.scope("parent") as parent:
        obs.record_request(parent, LLMRequest(model="test"))
        with obs.scope("child") as child:
            assert trace.get_current_span() is child
            assert child.get_span_context().trace_id == parent.get_span_context().trace_id
    project.assert_not_called()
    provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cooperative", [False, True])
async def test_cancelled_scope_propagates_without_error_status(cooperative: bool) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    obs = service(provider)
    error = IrisCancellationRequestedError("cancel") if cooperative else asyncio.CancelledError()
    with pytest.raises(type(error)) as caught, obs.scope("cancelled"):
        raise error
    assert caught.value is error
    assert exporter.get_finished_spans()[0].status.status_code == StatusCode.UNSET
    provider.shutdown()


@pytest.mark.parametrize(
    "kind,stage,status",
    [
        ("memory", "flush", "blocked"),
        ("memory", "dream", "completed"),
        ("memory", "overview", "failed"),
        ("evolution", "experience", "empty"),
        ("evolution", "revision", "failed"),
        ("evolution", "revision", "cancelled"),
    ],
)
def test_domain_result_only_marks_real_failure_on_current_cycle(
    observability: tuple[Observability, InMemorySpanExporter],
    kind: str,
    stage: str,
    status: str,
) -> None:
    obs, exporter = observability
    with obs.bind({"iris.maintenance.kind": kind}), obs.scope("iris.maintenance.cycle"):
        obs.maintenance_result(
            kind, stage, status, revision_id="revision" if stage == "revision" else None
        )
    [span] = exporter.get_finished_spans()
    [event] = span.events
    assert event.name == "iris.maintenance.result"
    assert event.attributes["iris.maintenance.kind"] == kind
    assert event.attributes["iris.maintenance.stage"] == stage
    assert event.attributes["iris.maintenance.status"] == status
    assert (span.status.status_code is StatusCode.ERROR) is (status == "failed")
    assert ("iris.maintenance.revision_id" in event.attributes) is (stage == "revision")


def test_standalone_domain_result_does_not_write_to_host_or_wrong_cycle(
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    obs, exporter = observability
    with obs.scope("host"):
        obs.maintenance_result("memory", "overview", "failed")
    with obs.bind({"iris.maintenance.kind": "evolution"}), obs.scope("iris.maintenance.cycle"):
        obs.maintenance_result("memory", "overview", "failed")
    assert all(not span.events for span in exporter.get_finished_spans())
    assert all(
        span.status.status_code is StatusCode.UNSET for span in exporter.get_finished_spans()
    )
