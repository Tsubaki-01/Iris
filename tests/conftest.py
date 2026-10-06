from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# 子目录 conftest 和测试模块导入前即使用随包模型表，子进程继承同一离线设置。
_litellm_environment = pytest.MonkeyPatch()
_litellm_environment.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from iris.observability.service import Observability

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config: pytest.Config) -> None:
    """测试会话退出时恢复宿主进程原有的 LiteLLM 环境设置。"""
    config.add_cleanup(_litellm_environment.undo)


@pytest.fixture
def otel_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """观测测试固定 SDK 行为，不继承宿主的禁用、采样与属性裁剪配置。"""
    for name in (
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_SAMPLER",
        "OTEL_TRACES_SAMPLER_ARG",
        "OTEL_ATTRIBUTE_COUNT_LIMIT",
        "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
        "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT",
        "OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT",
        "OTEL_SPAN_EVENT_COUNT_LIMIT",
        "OTEL_SPAN_LINK_COUNT_LIMIT",
        "OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT",
        "OTEL_LINK_ATTRIBUTE_COUNT_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def observability(
    otel_test_environment: None,
) -> Iterator[tuple[Observability, InMemorySpanExporter]]:
    """共享实际内存导出器；仅使用本 fixture 的测试加载可选 SDK。"""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
    from iris.observability.service import Observability

    provider = TracerProvider(sampler=ALWAYS_ON, shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    service = Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=True),
        ObservabilityExportConfig(),
        tracer_provider=provider,
    )
    yield service, exporter
    provider.shutdown()
