from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from iris.observability.service import Observability

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def observability() -> Iterator[tuple[Observability, InMemorySpanExporter]]:
    """共享实际内存导出器；仅使用本 fixture 的测试加载可选 SDK。"""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
    from iris.observability.service import Observability

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
