"""Lifecycle harness contract 共用的确定性 fixtures。"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability


@pytest.fixture
def lifecycle_now() -> datetime:
    """返回不会读取 wall clock 的固定 UTC 时间。"""
    return datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


@pytest.fixture
def lifecycle_later(lifecycle_now: datetime) -> datetime:
    """返回固定时间之后的一秒。"""
    return lifecycle_now + timedelta(seconds=1)


@pytest.fixture
def observability() -> Iterator[tuple[Observability, InMemorySpanExporter]]:
    """实际内存导出器，供 harness 的调用关系与共享关闭测试复用。"""
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    yield service, exporter
    provider.shutdown()
